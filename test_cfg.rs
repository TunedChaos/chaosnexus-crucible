use candle_transformers::models::llama::{LlamaConfig, Config};
fn main() {
    let json: LlamaConfig = serde_json::from_str("{}").unwrap();
    let cfg: Config = json.into_config(false);
}
