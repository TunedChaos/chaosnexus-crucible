// chaosnexus-crucible/src/engine/quantized_granite.rs
//! Quantized Granite GGUF loader for Candle.
//!
//! Candle's `quantized_llama::ModelWeights::from_gguf` only reads `llama.*`
//! metadata. IBM Granite GGUFs (including ChaosNexus Tuned v1) store
//! `granite.*` keys plus extra scales (`embedding_scale`, `residual_scale`,
//! `attention.scale`, `logit_scale`). This module mirrors the dense Llama
//! GGUF graph with those Granite-specific scales applied the way llama.cpp
//! does in `llama_model_granite`.

use std::collections::HashMap;
use std::io::{Read, Seek};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::{build_causal_mask, repeat_kv};

const MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Ok(Self {
            inner: candle_core::quantized::QMatMul::from_qtensor(qtensor)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Attention logits scale (`granite.attention.scale`), or `1/sqrt(head_dim)`.
    attention_scale: f64,
    residual_scale: f64,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // Granite uses interleaved (NORM) RoPE like Llama, not NEOX.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    (Tensor::cat(&[k_cache, &k], 2)?, Tensor::cat(&[v_cache, &v], 2)?)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;
        let att = (q.matmul(&k.t()?)? * self.attention_scale)?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attention_wo.forward(&y)
    }
}

/// Dense quantized Granite weights loaded from a GGUF file.
#[derive(Debug)]
pub struct GraniteModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    embedding_scale: f64,
    /// llama.cpp divides logits by this value (`1.0 / f_logit_scale`).
    logit_scale: f64,
    masks: HashMap<(usize, usize), Tensor>,
}

fn precomput_freqs_cis(head_dim: usize, freq_base: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

fn md_get<'a>(ct: &'a gguf_file::Content, key: &str) -> Result<&'a Value> {
    ct.metadata
        .get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("cannot find {key} in metadata")))
}

fn md_u32(ct: &gguf_file::Content, key: &str) -> Result<u32> {
    md_get(ct, key)?.to_u32()
}

fn md_f32(ct: &gguf_file::Content, key: &str) -> Result<f32> {
    md_get(ct, key)?.to_f32()
}

fn md_f32_or(ct: &gguf_file::Content, key: &str, default: f32) -> f32 {
    md_get(ct, key).ok().and_then(|v| v.to_f32().ok()).unwrap_or(default)
}

/// True when GGUF `general.architecture` is granite (dense).
pub fn is_granite_gguf(ct: &gguf_file::Content) -> bool {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.as_str() == "granite")
        .unwrap_or(false)
}

impl GraniteModelWeights {
    /// Load a dense Granite Q4/Q*_K GGUF (tensor names match Llama GGUF layout).
    pub fn from_gguf<R: Seek + Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let head_count = md_u32(&ct, "granite.attention.head_count")? as usize;
        let head_count_kv = md_u32(&ct, "granite.attention.head_count_kv")? as usize;
        let block_count = md_u32(&ct, "granite.block_count")? as usize;
        let embedding_length = md_u32(&ct, "granite.embedding_length")? as usize;
        let rope_dim = md_u32(&ct, "granite.rope.dimension_count")? as usize;
        let rms_norm_eps =
            md_f32(&ct, "granite.attention.layer_norm_rms_epsilon")? as f64;
        let rope_freq_base = md_f32_or(&ct, "granite.rope.freq_base", 10_000.0);
        let embedding_scale = md_f32_or(&ct, "granite.embedding_scale", 1.0) as f64;
        let residual_scale = md_f32_or(&ct, "granite.residual_scale", 1.0) as f64;
        let attention_scale = {
            let s = md_f32_or(&ct, "granite.attention.scale", 0.0);
            if s == 0.0 {
                1.0 / (rope_dim as f64).sqrt()
            } else {
                s as f64
            }
        };
        let logit_scale = md_f32_or(&ct, "granite.logit_scale", 1.0) as f64;
        if logit_scale == 0.0 {
            candle_core::bail!("granite.logit_scale must be non-zero");
        }

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let feed_forward_w1 =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_w2 =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let feed_forward_w3 =
                ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp: Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                },
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                attention_scale,
                residual_scale,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            embedding_scale,
            logit_scale,
            masks: HashMap::new(),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            return Ok(mask.clone());
        }
        let mask = build_causal_mask(seq_len, index_pos, device)?;
        self.masks.insert((seq_len, kv_len), mask.clone());
        Ok(mask)
    }

    /// Clear KV cache between independent prompts.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }

    /// Forward pass returning logits for the last token (scaled for Granite).
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let mut layer_in = self.tok_embeddings.forward(x)?;
        if self.embedding_scale != 1.0 {
            layer_in = (layer_in * self.embedding_scale)?;
        }

        for layer in self.layers.iter_mut() {
            let residual = layer_in.clone();
            let x = layer.attention_norm.forward(&layer_in)?;
            let mut attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            if layer.residual_scale != 1.0 {
                attn = (attn * layer.residual_scale)?;
            }
            let x = (attn + residual)?;

            let residual = x.clone();
            let x = layer.ffn_norm.forward(&x)?;
            let mut ffn = layer.mlp.forward(&x)?;
            if layer.residual_scale != 1.0 {
                ffn = (ffn * layer.residual_scale)?;
            }
            layer_in = (ffn + residual)?;
        }

        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.output.forward(&x)?;
        // llama.cpp: `ggml_scale(cur, 1.0f / hparams.f_logit_scale)`
        logits / self.logit_scale
    }
}
