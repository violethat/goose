//! Bounds how many images are re-sent to the provider each turn.
//!
//! Images ride in the conversation as base64 blocks (tool results from
//! `read_image`, user attachments). Every request re-sends all of them, so a
//! long session with many image calls grows the request body until gateways
//! reject it with "request body is too large" long before the token budget
//! is exhausted. Only the most recent images are relevant for continuity;
//! older ones are replaced with a short text placeholder. Their source path
//! remains in the adjacent tool-result text, so the model can re-read an old
//! image when it actually needs to see it again.

use crate::conversation::message::{Message, MessageContentBlock, ToolResponse};
use rmcp::model::{CallToolResult, ContentBlock};
use std::borrow::Cow;

type ToolResult<T> = Result<T, rmcp::model::ErrorData>;

const DEFAULT_MAX_RETAINED_IMAGES: usize = 4;
const PLACEHOLDER: &str =
    "[image omitted from history: only the most recent images are re-sent to the model]";

/// Number of images kept in outgoing requests. Older images become text
/// placeholders. `0` keeps none; a large value effectively disables pruning.
pub fn max_retained_images() -> usize {
    std::env::var("GOOSE_MAX_RETAINED_IMAGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_RETAINED_IMAGES)
}

fn images_in_message(message: &Message) -> usize {
    message
        .content
        .iter()
        .map(|block| match block {
            MessageContentBlock::Image(_) => 1,
            MessageContentBlock::ToolResponse(response) => match &response.tool_result {
                ToolResult::Ok(result) => result
                    .content
                    .iter()
                    .filter(|c| matches!(c, ContentBlock::Image(_)))
                    .count(),
                ToolResult::Err(_) => 0,
            },
            _ => 0,
        })
        .sum()
}

/// Replaces every image except the most recent `max_retained_images()` ones
/// with a text placeholder. Returns borrowed input when nothing changes.
pub fn retain_recent_images(messages: &[Message]) -> Cow<'_, [Message]> {
    let limit = max_retained_images();
    let total: usize = messages.iter().map(images_in_message).sum();
    if total <= limit {
        return Cow::Borrowed(messages);
    }
    let images_to_drop = total - limit;

    let mut dropped = 0usize;
    let mut retained: Vec<Message> = Vec::with_capacity(messages.len());
    for message in messages {
        let mut message = message.clone();
        message.content = std::mem::take(&mut message.content)
            .into_iter()
            .map(|block| match block {
                MessageContentBlock::Image(content) => {
                    if dropped < images_to_drop {
                        dropped += 1;
                        MessageContentBlock::Text(rmcp::model::TextContent::new(PLACEHOLDER))
                    } else {
                        MessageContentBlock::Image(content)
                    }
                }
                MessageContentBlock::ToolResponse(mut response) => {
                    prune_tool_response(&mut response, &mut dropped, images_to_drop);
                    MessageContentBlock::ToolResponse(response)
                }
                other => other,
            })
            .collect();
        retained.push(message);
    }
    Cow::Owned(retained)
}

fn prune_tool_response(response: &mut ToolResponse, dropped: &mut usize, images_to_drop: usize) {
    if *dropped >= images_to_drop {
        return;
    }
    let ToolResult::Ok(result) = &mut response.tool_result else {
        return;
    };
    if !result
        .content
        .iter()
        .any(|c| matches!(c, ContentBlock::Image(_)))
    {
        return;
    }
    let mut new_call_result = CallToolResult::success(Vec::new());
    for content in std::mem::take(&mut result.content) {
        match content {
            ContentBlock::Image(_) if *dropped < images_to_drop => {
                *dropped += 1;
                new_call_result
                    .content
                    .push(ContentBlock::text(PLACEHOLDER));
            }
            other => new_call_result.content.push(other),
        }
    }
    response.tool_result = ToolResult::Ok(new_call_result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::{Message, MessageContentBlock};
    use rmcp::model::ImageContent;

    fn image_block() -> MessageContentBlock {
        MessageContentBlock::Image(ImageContent::new("aGk=", "image/png"))
    }

    fn user_with_image() -> Message {
        Message::user().with_content(image_block())
    }

    fn tool_result_with_image(id: &str) -> Message {
        Message::user().with_tool_response(
            id,
            Ok(CallToolResult::success(vec![
                ContentBlock::text("Loaded image from /tmp/shot.png"),
                ContentBlock::Image(ImageContent::new("aGk=", "image/png")),
            ])),
        )
    }

    fn count_placeholders(messages: &[Message]) -> usize {
        messages
            .iter()
            .flat_map(|m| m.content.iter())
            .map(|b| match b {
                MessageContentBlock::Text(t) if t.text == PLACEHOLDER => 1,
                MessageContentBlock::ToolResponse(response) => match &response.tool_result {
                    ToolResult::Ok(result) => result
                        .content
                        .iter()
                        .filter(|c| c.as_text().is_some_and(|t| t.text == PLACEHOLDER))
                        .count(),
                    ToolResult::Err(_) => 0,
                },
                _ => 0,
            })
            .sum()
    }

    fn count_images(messages: &[Message]) -> usize {
        messages.iter().map(images_in_message).sum()
    }

    #[test]
    fn keeps_all_images_when_under_the_limit() {
        let _guard = env_lock::lock_env([("GOOSE_MAX_RETAINED_IMAGES", None::<&str>)]);
        let messages: Vec<Message> = (0..3).map(|_| user_with_image()).collect();
        let retained = retain_recent_images(&messages);
        assert!(matches!(retained, Cow::Borrowed(_)));
        assert_eq!(count_images(&retained), 3);
    }

    #[test]
    fn drops_oldest_images_beyond_the_limit() {
        let _guard = env_lock::lock_env([("GOOSE_MAX_RETAINED_IMAGES", Some("2"))]);
        let messages: Vec<Message> = (0..5).map(|_| user_with_image()).collect();
        let retained = retain_recent_images(&messages);
        assert!(matches!(retained, Cow::Owned(_)));
        assert_eq!(count_images(&retained), 2);
        assert_eq!(count_placeholders(&retained), 3);
        // The two kept images are the newest ones.
        assert!(matches!(
            retained[4].content[0],
            MessageContentBlock::Image(_)
        ));
        assert!(matches!(
            retained[3].content[0],
            MessageContentBlock::Image(_)
        ));
        assert_eq!(count_placeholders(&[retained[0].clone()]), 1);
    }

    #[test]
    fn prunes_images_inside_tool_responses() {
        let _guard = env_lock::lock_env([("GOOSE_MAX_RETAINED_IMAGES", Some("1"))]);
        let messages = vec![
            tool_result_with_image("req_1"),
            tool_result_with_image("req_2"),
        ];
        let retained = retain_recent_images(&messages);
        assert_eq!(count_images(&retained), 1);
        assert_eq!(count_placeholders(&retained), 1);
        // The tool result keeps its text content alongside the placeholder.
        let MessageContentBlock::ToolResponse(response) = &retained[0].content[0] else {
            panic!("expected tool response");
        };
        let ToolResult::Ok(result) = &response.tool_result else {
            panic!("expected successful tool result");
        };
        assert_eq!(result.content.len(), 2);
        assert_eq!(
            result.content[0].as_text().map(|t| t.text.as_str()),
            Some("Loaded image from /tmp/shot.png")
        );
    }

    #[test]
    fn zero_limit_removes_every_image() {
        let _guard = env_lock::lock_env([("GOOSE_MAX_RETAINED_IMAGES", Some("0"))]);
        let messages = vec![user_with_image(), tool_result_with_image("req_1")];
        let retained = retain_recent_images(&messages);
        assert_eq!(count_images(&retained), 0);
        assert_eq!(count_placeholders(&retained), 2);
    }
}
