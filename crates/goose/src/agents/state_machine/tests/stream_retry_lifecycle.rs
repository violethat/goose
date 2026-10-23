use anyhow::Result;

use super::dummy_api::ProviderFeatures;
use super::pipeline::MessageKind::Agent;
use super::pipeline::test_pipeline_with;
use crate::conversation::message::Message;

#[tokio::test]
async fn mid_stream_transient_error_restarts_the_request() -> Result<()> {
    let (pipeline, api) = test_pipeline_with(ProviderFeatures::default()).await?;

    // Registered first so it is only reached after the error rule is spent
    // (rules match in reverse registration order).
    api.on("hello").reply("recovered after restart");
    api.on("hello")
        .reply("partial")
        .times(1)
        .server_error("upstream dropped the stream");

    let result = pipeline.run_message(Message::user().with_text("hello")).await?;

    result.assert_message(-1, Agent, "recovered after restart");
    assert_eq!(
        api.call_count(),
        2,
        "the interrupted turn must be retried with a fresh request"
    );

    Ok(())
}

#[tokio::test]
async fn transient_retries_are_bounded() -> Result<()> {
    let (pipeline, api) = test_pipeline_with(ProviderFeatures::default()).await?;

    // Every attempt streams a little content and then dies with a 5xx. After
    // MAX_STREAM_INTERRUPT_RETRIES restarts the turn must give up with an
    // error instead of looping forever.
    api.on("hello")
        .reply("stalling")
        .server_error("upstream is on fire");

    let result = pipeline.run_message(Message::user().with_text("hello")).await?;

    // 1 initial attempt + 3 restarts.
    assert_eq!(api.call_count(), 4);
    let last = result.conversation().messages().last().unwrap();
    assert!(
        last.error_kind().is_some(),
        "exhausted restarts must surface an error"
    );

    Ok(())
}
