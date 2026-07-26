# Security Policy & Architecture

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x (Current) | :white_check_mark: |
| < 0.1.0 | :x: |

---

## Security Model & Sandboxing Architecture

ChaosNexus is built ground-up for secure, deterministic execution of autonomous AI agent plugins and tools. To prevent untrusted scripts or autonomous agent breakout, ChaosNexus enforces a **Defense-in-Depth** security model comprising six distinct layers:

### 1. In-Memory Rhai Script Engine Sandboxing
All plugin logic in **ChaosNexus Anvil** runs within an embedded [Rhai](https://rhai.rs/) engine. Rhai is strictly sandboxed in memory:
- Rhai scripts have **no ambient access** to host filesystem, network, system environment, or process execution.
- Rhai scripts cannot invoke native system calls or load dynamic binary modules (`.so` / `.dll`).

### 2. Host-Authoritative Default-Deny Capability Model
Capability grants (e.g. `fs`, `shell`, `net`, `env`, `install`) are controlled **exclusively** by the host configuration file (`chaosnexus-anvil.toml` or `config.toml`):
- Plugin manifests (`plugin.toml`) **cannot self-authorize** permissions.
- If a plugin is unconfigured in `chaosnexus-anvil.toml`, it receives **zero capabilities** (`default-deny`).
- A plugin cannot inspect or access another plugin's configuration or secret keys (`CONFIG.secrets` is strictly identity-scoped).

### 3. Subdomain-Aware Egress Domain Allowlisting (`net_allowlist`)
To prevent autonomous agents from making unauthorized network egress calls or accessing arbitrary external infrastructure:
- Outbound HTTP, WebSocket, and MCP client calls validate destination hosts against `permissions.net_allowlist` (or `http_domains`).
- Wildcard subdomain patterns (e.g. `*.github.com`) are supported; un-allowlisted network calls raise immediate security violations.

### 4. Primary Command Gating & Parameterized Execution (`permissions.shell`)
System process execution via `run_command` is strictly tokenized and allowlisted:
- `shell_exec.rs` extracts the primary command executable (e.g., `"echo"`) and verifies it against the host's `permissions.shell` allowlist.
- Shell sub-interpreters (`/bin/sh -c`) are avoided in favor of array-based argument vectors `run_command(exec, ["arg1", "arg2"])` to prevent shell metacharacter injection.

### 5. Linux Host Process Isolation (`bwrap` Containerization)
When **ChaosNexus Forge** launches external CLI agent binaries (such as Goose CLI or AGY CLI) via `cli_agent_bridge`:
- If `bwrap` (bubblewrap) is present on Linux, child processes are wrapped in lightweight container namespaces restricting filesystem mounts (`--ro-bind / /`, `--bind <workspace> <workspace>`), preventing external agent processes from accessing host paths outside the designated workspace directory.

### 6. Distributed Trace Hop Bounds & Anti-Recursion Controls
To prevent self-amplifying sub-agent loops or infinite tool cascades:
- All execution paths carry an OpenTelemetry-compatible `trace_id`.
- Execution propagation checks a strict maximum hop limit (`MAX_HOP_COUNT = 5`). If an agent attempts to chain sub-agent tool calls beyond 5 hops, the execution engine aborts immediately.

---

## Reporting a Vulnerability

We take the security of ChaosNexus very seriously. If you discover a vulnerability, security flaw, or potential sandbox breakout, please report it responsibly:

1. **Do NOT open a public issue** on Codeberg, GitHub, or elsewhere for undisclosed security vulnerabilities.
2. Email your findings directly to **security@tnd.cx** or send an encrypted message to the project maintainers.
3. Include detailed steps to reproduce the issue, along with any relevant script snippets or proof-of-concept code.
4. The maintainers will acknowledge receipt within 48 hours and work with you to remediate and publish a fix.

Public discussion of already-fixed issues may use Codeberg (contribution primary). GitHub mirrors are read-only.

We appreciate the efforts of security researchers and open-source contributors in keeping ChaosNexus secure.

Background on why we treat agent sandboxing as non-optional: https://chaosnexus.ai/guide/why-local-sandboxing
