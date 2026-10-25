use goose_provider_types::base::Provider;
use goose_provider_types::model::ModelConfig;
use goose_providers::openai::from_declarative_config;
use rmcp::model::Tool;
use rmcp::object;
use serde_json::json;
use std::collections::HashMap;
use wiremock::matchers::{body_partial_json, body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn qubax_config_with_base(base: &str) -> goose_providers::declarative::DeclarativeProviderConfig {
    let mut config = goose_providers::declarative::deserialize_provider_config(include_str!(
        "../src/declarative/definitions/qubax.json"
    ))
    .unwrap();
    config.base_url = base.to_string();
    config.api_key_env = String::new();
    config.requires_auth = false;
    config
}

fn sol_model() -> ModelConfig {
    let mut model = ModelConfig::new("gpt-6.1-sol");
    model.reasoning = Some(true);
    model.with_merged_request_params(HashMap::from([(
        "thinking_effort".to_string(),
        json!("medium"),
    )]))
}

fn tool() -> Tool {
    Tool::new(
        "t",
        "d",
        object!({"type": "object", "properties": {}, "required": []}),
    )
}

const RESPONSES_SSE: &str = concat!(
    r#"data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"m1","output_index":0,"content_index":0,"delta":"ok"}"#,
    "\n\n",
    r#"data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","object":"response","created_at":0,"status":"completed","model":"gpt-6.1-sol","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
    "\n\ndata: [DONE]\n\n",
);

const CHAT_SSE: &str = concat!(
    "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
    "data: [DONE]\n\n"
);

fn effort_400() -> serde_json::Value {
    json!({
        "error": {"message": "Function tools with reasoning_effort are not supported for gpt-6.1-sol in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'."}
    })
}

#[tokio::test]
async fn qubax_routes_reasoning_models_to_the_responses_api() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(effort_400()))
        .expect(0)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(
            json!({"model": "gpt-6.1-sol", "reasoning": {"effort": "medium"}}),
        ))
        .and(body_string_contains("\"type\":\"function\""))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_string(RESPONSES_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let config = qubax_config_with_base(&server.uri());
    let provider = from_declarative_config(
        config,
        None,
        goose_providers::declarative::EnvKeyResolver {},
    )
    .unwrap()
    .build();

    let stream = provider
        .stream(&sol_model(), "be brief", &[], &[tool()])
        .await
        .expect("gpt-6.1-sol should stream via the responses API");
    let (message, _) = goose_provider_types::base::collect_stream(stream)
        .await
        .unwrap();
    assert_eq!(message.as_concat_text(), "ok");
    server.verify().await;
}

#[tokio::test]
async fn qubax_chat_completions_drops_reasoning_effort_after_rejection() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("reasoning_effort"))
        .respond_with(ResponseTemplate::new(400).set_body_json(effort_400()))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_string(CHAT_SSE),
        )
        .mount(&server)
        .await;

    let mut config = qubax_config_with_base(&server.uri());
    config.disable_responses_api = true;
    let provider = from_declarative_config(
        config,
        None,
        goose_providers::declarative::EnvKeyResolver {},
    )
    .unwrap()
    .build();

    let stream = provider
        .stream(&sol_model(), "be brief", &[], &[tool()])
        .await
        .expect("chat-completions should recover by dropping reasoning_effort");
    let (message, _) = goose_provider_types::base::collect_stream(stream)
        .await
        .unwrap();
    assert_eq!(message.as_concat_text(), "ok");
    server.verify().await;
}
