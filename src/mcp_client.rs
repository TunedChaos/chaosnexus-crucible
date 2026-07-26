use rust_mcp_sdk::{
    mcp_client::{ClientRuntime, McpClientOptions, ClientHandler, client_runtime::create_client},
    schema::{ClientCapabilities, Implementation, InitializeRequestParams},
    ToMcpClientHandler, McpClient,
};
use rust_mcp_transport::{ClientSseTransport, ClientSseTransportOptions};
use std::sync::Arc;

pub struct CrucibleClientHandler;

#[async_trait::async_trait]
impl ClientHandler for CrucibleClientHandler {
    // Relying on default implementations for ping, etc.
}

pub async fn init_mcp_client(port: u16, token: &str) -> Result<Arc<ClientRuntime>, String> {
    let url = format!("http://127.0.0.1:{}/sse?token={}", port, token);
    let transport = ClientSseTransport::new(&url, ClientSseTransportOptions::default())
        .map_err(|e| format!("Failed to create SSE transport: {}", e))?;

    let client_details = InitializeRequestParams {
        protocol_version: "2024-11-05".to_string(), // latest standard
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "ChaosNexus Crucible".to_string(),
            version: "0.1.0".to_string(),
            description: None,
            icons: vec![],
            title: None,
            website_url: None,
        },
        meta: None,
    };

    let options = McpClientOptions {
        client_details,
        transport,
        handler: CrucibleClientHandler.to_mcp_client_handler(),
        task_store: None,
        server_task_store: None,
        message_observer: None,
    };

    let client = create_client(options);
    
    // Start the client runtime
    client.clone().start().await.map_err(|e| format!("Failed to start MCP client: {}", e))?;

    Ok(client)
}
