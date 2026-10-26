use std::time::Duration;

use crate::agents::tool_execution::ToolCallContext;
use async_trait::async_trait;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool,
};
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};

pub static EXTENSION_NAME: &str = "exa";

const MAX_RESULTS: u32 = 10;
const MAX_TEXT_CHARS: usize = 1200;

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchParams {
    /// Natural language search query. Exa's neural search understands
    /// meaning, not just keywords, so phrase what you want to find.
    query: String,
    /// How many results to return (1-10).
    #[serde(default)]
    num_results: Option<u32>,
    /// "auto" (recommended), "neural" (semantic), or "keyword" (literal).
    #[serde(default)]
    search_type: Option<String>,
    /// Restrict to a category such as "research paper", "news",
    /// "github", "company", "personal site", "pdf", or "linkedin profile".
    #[serde(default)]
    category: Option<String>,
    /// Only include results from these domains.
    #[serde(default)]
    include_domains: Option<Vec<String>>,
    /// Exclude results from these domains.
    #[serde(default)]
    exclude_domains: Option<Vec<String>>,
    /// ISO date (2024-01-01 or full timestamp): only results published after this.
    #[serde(default)]
    start_published_date: Option<String>,
    /// ISO date: only results published before this.
    #[serde(default)]
    end_published_date: Option<String>,
    /// Include each page's cleaned main text in the results.
    #[serde(default)]
    include_text: Option<bool>,
    /// Include Exa's generated one-paragraph summary of each page.
    #[serde(default)]
    include_summary: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetContentsParams {
    /// URLs or result IDs from a previous exa search to fetch the contents of.
    ids: Vec<String>,
    /// Include the cleaned main text, truncated to a readable length.
    #[serde(default)]
    include_text: Option<bool>,
    /// Include Exa's generated summary.
    #[serde(default)]
    include_summary: Option<bool>,
    /// Include key highlights (relevant excerpt quotes) from the page.
    #[serde(default)]
    include_highlights: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FindSimilarParams {
    /// URL of a page to find similar pages for.
    url: String,
    /// How many results to return (1-10).
    #[serde(default)]
    num_results: Option<u32>,
}

pub struct ExaClient {
    info: InitializeResult,
}

impl ExaClient {
    pub fn new(_context: PlatformExtensionContext) -> anyhow::Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Exa Search"),
            );

        Ok(Self { info })
    }

    fn get_tools() -> Vec<Tool> {
        let search_schema =
            serde_json::to_value(schema_for!(SearchParams)).expect("schema serialization");
        let contents_schema =
            serde_json::to_value(schema_for!(GetContentsParams)).expect("schema serialization");
        let similar_schema =
            serde_json::to_value(schema_for!(FindSimilarParams)).expect("schema serialization");

        vec![
            Tool::new(
                "search",
                "High-quality web search over Exa's neural index. Prefer this over free search \
                 engines for research, technical topics, and anything where result quality \
                 matters. Returns titles, URLs, dates, and optionally page text and summaries.",
                search_schema.as_object().unwrap().clone(),
            ),
            Tool::new(
                "get_contents",
                "Fetch the cleaned content of web pages by URL or by ID returned from an exa \
                 search: full text, summary, and/or key highlights.",
                contents_schema.as_object().unwrap().clone(),
            ),
            Tool::new(
                "find_similar",
                "Find pages similar to a given URL. Useful to expand from one good source to \
                 the neighborhood around it.",
                similar_schema.as_object().unwrap().clone(),
            ),
        ]
    }

    async fn api_key(&self) -> Result<String, String> {
        crate::config::Config::global()
            .get_secret::<String>("EXA_API_KEY")
            .map_err(|_| {
                "EXA_API_KEY is not configured. Add it with `goose configure` or set the \
                 EXA_API_KEY environment variable."
                    .to_string()
            })
    }
}

fn error_result(message: impl std::fmt::Display) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(format!("Error: {message}"))])
}

fn text_result(message: String) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(message)])
}

async fn exa_post(endpoint: &str, api_key: &str, payload: Value) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .user_agent(concat!(
            "goose/",
            env!("CARGO_PKG_VERSION"),
            " (+https://github.com/aaif-goose/goose)"
        ))
        .timeout(Duration::from_secs(45))
        .build()
        .map_err(|error| format!("failed to build HTTP client: {error}"))?;

    let response = client
        .post(format!("https://api.exa.ai/{endpoint}"))
        .header("x-api-key", api_key)
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("request to exa failed: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("failed to read exa response: {error}"))?;

    if !status.is_success() {
        let detail = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("message")
                    .or_else(|| value.get("error"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| body.chars().take(300).collect());
        return Err(format!("exa returned {}: {detail}", status.as_u16()));
    }

    serde_json::from_str(&body).map_err(|error| format!("failed to parse exa response: {error}"))
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut cut: String = text.chars().take(max_chars).collect();
    cut.push('…');
    cut
}

fn render_results(
    results: &[Value],
    include_text: bool,
    include_summary: bool,
    include_highlights: bool,
) -> String {
    let mut rendered = Vec::with_capacity(results.len());
    for (index, result) in results.iter().enumerate() {
        let title = result
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("(untitled)");
        let url = result.get("url").and_then(Value::as_str).unwrap_or("");
        let published = result
            .get("publishedDate")
            .and_then(Value::as_str)
            .map(|date| format!(" · published {date}"))
            .filter(|s| !s.contains("0001-01-01"))
            .unwrap_or_default();
        let mut entry = format!("[{}] {} — {}{}", index + 1, title, url, published);
        if include_summary {
            if let Some(summary) = result.get("summary").and_then(Value::as_str) {
                entry.push_str(&format!("\n    summary: {}", truncate(summary, 500)));
            }
        }
        if include_text {
            if let Some(text) = result.get("text").and_then(Value::as_str) {
                entry.push_str(&format!("\n    text: {}", truncate(text, MAX_TEXT_CHARS)));
            }
        }
        if include_highlights {
            if let Some(highlights) = result
                .get("highlights")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .take(3)
                        .collect::<Vec<_>>()
                })
                .filter(|items| !items.is_empty())
            {
                entry.push_str(&format!("\n    highlights: {}", highlights.join(" ⸱ ")));
            }
        }
        rendered.push(entry);
    }
    rendered.join("\n\n")
}

fn clamp_results(requested: Option<u32>) -> u32 {
    requested.unwrap_or(5).clamp(1, MAX_RESULTS)
}

#[async_trait]
impl McpClientTrait for ExaClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        _cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let api_key = match self.api_key().await {
            Ok(key) => key,
            Err(message) => return Ok(error_result(message)),
        };
        let _ = ctx.working_dir_str();

        match name {
            "search" => {
                let params: SearchParams = match parse_arguments(arguments) {
                    Ok(params) => params,
                    Err(message) => return Ok(error_result(message)),
                };
                let include_text = params.include_text.unwrap_or(false);
                let include_summary = params.include_summary.unwrap_or(true);
                let mut payload = json!({
                    "query": params.query,
                    "numResults": clamp_results(params.num_results),
                    "type": params.search_type.as_deref().unwrap_or("auto"),
                });
                if let Some(category) = params.category {
                    payload["category"] = json!(category);
                }
                if let Some(domains) = params.include_domains {
                    payload["includeDomains"] = json!(domains);
                }
                if let Some(domains) = params.exclude_domains {
                    payload["excludeDomains"] = json!(domains);
                }
                if let Some(date) = params.start_published_date {
                    payload["startPublishedDate"] = json!(date);
                }
                if let Some(date) = params.end_published_date {
                    payload["endPublishedDate"] = json!(date);
                }
                if include_text {
                    payload["contents"] = json!({ "text": true });
                }
                let response = match exa_post("search", &api_key, payload).await {
                    Ok(response) => response,
                    Err(message) => return Ok(error_result(message)),
                };
                let results = response
                    .get("results")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if results.is_empty() {
                    return Ok(text_result("No results found for that query.".to_string()));
                }
                Ok(text_result(render_results(
                    &results,
                    include_text,
                    include_summary,
                    false,
                )))
            }
            "get_contents" => {
                let params: GetContentsParams = match parse_arguments(arguments) {
                    Ok(params) => params,
                    Err(message) => return Ok(error_result(message)),
                };
                if params.ids.is_empty() {
                    return Ok(error_result("ids must not be empty"));
                }
                let include_text = params.include_text.unwrap_or(true);
                let include_summary = params.include_summary.unwrap_or(true);
                let include_highlights = params.include_highlights.unwrap_or(false);
                let mut contents = json!({});
                if include_text {
                    contents["text"] = json!({ "maxCharacters": 5000 });
                }
                if include_summary {
                    contents["summary"] = json!(true);
                }
                if include_highlights {
                    contents["highlights"] = json!({ "numSentences": 3 });
                }
                let payload = json!({
                    "ids": params.ids,
                    "contents": contents,
                });
                let response = match exa_post("contents", &api_key, payload).await {
                    Ok(response) => response,
                    Err(message) => return Ok(error_result(message)),
                };
                let results = response
                    .get("results")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if results.is_empty() {
                    return Ok(text_result("No content found for those IDs.".to_string()));
                }
                Ok(text_result(render_results(
                    &results,
                    include_text,
                    include_summary,
                    include_highlights,
                )))
            }
            "find_similar" => {
                let params: FindSimilarParams = match parse_arguments(arguments) {
                    Ok(params) => params,
                    Err(message) => return Ok(error_result(message)),
                };
                let payload = json!({
                    "url": params.url,
                    "numResults": clamp_results(params.num_results),
                    "contents": { "text": { "maxCharacters": 300 } },
                });
                let response = match exa_post("findSimilar", &api_key, payload).await {
                    Ok(response) => response,
                    Err(message) => return Ok(error_result(message)),
                };
                let results = response
                    .get("results")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if results.is_empty() {
                    return Ok(text_result("No similar pages found.".to_string()));
                }
                Ok(text_result(render_results(&results, false, true, false)))
            }
            other => Ok(error_result(format!("Unknown tool: {other}"))),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

fn parse_arguments<T: for<'de> Deserialize<'de>>(
    arguments: Option<JsonObject>,
) -> Result<T, String> {
    let value = Value::Object(arguments.unwrap_or_default().into_iter().collect());
    serde_json::from_value(value).map_err(|error| format!("invalid arguments: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_json(title: &str, url: &str, text: Option<&str>, summary: Option<&str>) -> Value {
        let mut value =
            json!({"title": title, "url": url, "publishedDate": "2025-06-01T00:00:00.000Z"});
        if let Some(text) = text {
            value["text"] = json!(text);
        }
        if let Some(summary) = summary {
            value["summary"] = json!(summary);
        }
        value
    }

    #[test]
    fn render_results_includes_text_and_summary_when_requested() {
        let results = vec![
            result_json(
                "A",
                "https://a.example",
                Some("alpha content"),
                Some("alpha summary"),
            ),
            result_json("B", "https://b.example", None, None),
        ];
        let rendered = render_results(&results, true, true, false);
        assert!(rendered.contains("[1] A — https://a.example · published 2025-06-01"));
        assert!(rendered.contains("summary: alpha summary"));
        assert!(rendered.contains("text: alpha content"));
        assert!(rendered.contains("[2] B — https://b.example"));
    }

    #[test]
    fn render_results_truncates_long_text() {
        let long = "x".repeat(5000);
        let results = vec![result_json("A", "https://a.example", Some(&long), None)];
        let rendered = render_results(&results, true, false, false);
        assert!(rendered.chars().count() < 2000);
        assert!(rendered.contains('…'));
    }

    #[test]
    fn render_results_includes_highlights() {
        let mut result = result_json("A", "https://a.example", None, None);
        result["highlights"] = json!(["first quote", "second quote"]);
        let rendered = render_results(&[result], false, false, true);
        assert!(rendered.contains("highlights: first quote ⸱ second quote"));
    }

    #[test]
    fn clamp_results_bounds_the_request() {
        assert_eq!(clamp_results(None), 5);
        assert_eq!(clamp_results(Some(0)), 1);
        assert_eq!(clamp_results(Some(99)), 10);
    }

    #[test]
    fn search_params_parse_from_partial_json() {
        let mut arguments = JsonObject::default();
        arguments.insert("query".to_string(), json!("rust async"));
        let params: SearchParams =
            parse_arguments(Some(arguments)).expect("only query is required");
        assert_eq!(params.query, "rust async");
        assert!(params.num_results.is_none());
        assert!(params.include_domains.is_none());
    }
}
