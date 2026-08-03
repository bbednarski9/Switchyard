// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::LlmExecutionNextFn;
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::error::FlowError;
use nemo_relay::observability::otel::{OpenTelemetrySubscriber, OpenTelemetrySubscriberOptions};
use nemo_relay::observability::{MarkProjection, OpenTelemetryType};
use nemo_relay::plugin::PluginConfig;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider, SpanData};
use serde_json::{Map, Value as Json, json};

const EVENT_SUBSCRIBER_NAME: &str = "switchyard_embedded_host_probe";
const OPENINFERENCE_SUBSCRIBER_NAME: &str = "switchyard_embedded_openinference_probe";
const AGENT_SCOPE_NAME: &str = "switchyard.embedded.agent";
const CALLER_CREDENTIAL: &str = "Bearer embedded-caller-secret";
const TARGET_CREDENTIAL: &str = "Bearer target-e2e";
const TARGET_CREDENTIAL_ENV: &str = "SWITCHYARD_EMBEDDED_TARGET_AUTHORIZATION";

fn argument(name: &str) -> Result<String, Box<dyn Error>> {
    std::env::args()
        .nth(match name {
            "manifest" => 1,
            "provider URL" => 2,
            _ => unreachable!("known argument name"),
        })
        .ok_or_else(|| {
            format!("missing {name}; usage: embedded-host <manifest> <provider-url>").into()
        })
}

fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == key)
        .map(|attribute| attribute.value.to_string())
}

fn assert_openinference_telemetry(spans: &[SpanData]) -> Result<(), Box<dyn Error>> {
    let agent_span = spans
        .iter()
        .find(|span| span.name.as_ref() == AGENT_SCOPE_NAME)
        .ok_or("missing completed OpenInference Agent span")?;
    if agent_span.start_time >= agent_span.end_time {
        return Err("OpenInference Agent span did not have a complete lifetime".into());
    }
    if attribute(agent_span, "openinference.span.kind").as_deref() != Some("AGENT") {
        return Err("embedded scope was not exported as an OpenInference AGENT".into());
    }

    let llm_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "openai.chat_completions")
        .ok_or("missing completed OpenInference LLM span")?;
    if llm_span.start_time >= llm_span.end_time {
        return Err("OpenInference LLM span did not have a complete lifetime".into());
    }
    for (key, expected) in [
        ("openinference.span.kind", "LLM"),
        ("llm.model_name", "fake/embedded"),
        ("input.value", "user: embedded host probe"),
        ("output.value", "chat from fake/embedded"),
        ("llm.token_count.prompt", "4"),
        ("llm.token_count.completion", "2"),
        ("llm.token_count.total", "6"),
    ] {
        let actual = attribute(llm_span, key);
        if actual.as_deref() != Some(expected) {
            return Err(format!(
                "OpenInference LLM span attribute {key:?} was {actual:?}, expected {expected:?}"
            )
            .into());
        }
    }

    let decision_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:switchyard.routing.decision")
        .ok_or("missing OpenInference Switchyard routing-decision span")?;
    let trace_id = agent_span.span_context.trace_id();
    if llm_span.span_context.trace_id() != trace_id
        || decision_span.span_context.trace_id() != trace_id
        || llm_span.parent_span_id != agent_span.span_context.span_id()
        || decision_span.parent_span_id != agent_span.span_context.span_id()
    {
        return Err(
            "OpenInference LLM and routing-decision spans were not sibling children of the Agent span"
                .into(),
        );
    }
    for (key, expected) in [
        ("openinference.span.kind", "TOOL"),
        ("tool.name", "switchyard.routing.decision"),
        ("nemo_relay.mark.data.algorithm", "random"),
        ("nemo_relay.mark.data.selected_target", "embedded"),
    ] {
        let actual = attribute(decision_span, key);
        if actual.as_deref() != Some(expected) {
            return Err(format!(
                "routing-decision span attribute {key:?} was {actual:?}, expected {expected:?}"
            )
            .into());
        }
    }
    if attribute(decision_span, "nemo_relay.mark.orphan").is_some() {
        return Err("Switchyard routing-decision span was marked as orphaned".into());
    }
    let agent_uuid = attribute(agent_span, "nemo_relay.uuid");
    let llm_parent_uuid = attribute(llm_span, "nemo_relay.parent_uuid");
    let decision_parent_uuid = attribute(decision_span, "nemo_relay.mark.parent_uuid");
    if llm_parent_uuid != agent_uuid || decision_parent_uuid != agent_uuid {
        return Err(
            "LLM and routing-decision event parent UUIDs do not match the Agent scope UUID".into(),
        );
    }

    let rendered = format!("{spans:?}");
    if rendered.contains(CALLER_CREDENTIAL) || rendered.contains(TARGET_CREDENTIAL) {
        return Err("OpenInference telemetry exposed a caller or target credential".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // SAFETY: this is the first operation in the single-threaded process,
    // before Tokio or the dynamic plugin can create another thread.
    unsafe {
        std::env::set_var(TARGET_CREDENTIAL_ENV, TARGET_CREDENTIAL);
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(argument("manifest")?).canonicalize()?;
    let provider_url = argument("provider URL")?.trim_end_matches('/').to_string();
    let plugin_config = json!({
        "version": 2,
        "priority": 0,
        "max_retries": 0,
        "algorithm": {
            "kind": "random",
            "seed": 42
        },
        "default_targets": {
            "openai_chat": "embedded"
        },
        "targets": {
            "embedded": {
                "model": "fake/embedded",
                "protocol": "openai_chat",
                "base_url": format!("{provider_url}/v1"),
                "weight": 1,
                "headers": {
                    "x-switchyard-target": "same"
                },
                "header_env": {
                    "authorization": TARGET_CREDENTIAL_ENV
                }
            }
        }
    });
    let config = plugin_config
        .as_object()
        .cloned()
        .ok_or("embedded-host plugin config must be an object")?;

    let (activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: "nvidia.switchyard".into(),
            kind: DynamicPluginKind::RustDynamic,
            manifest_ref: manifest.to_string_lossy().into_owned(),
            environment_ref: None,
            config,
        }],
    )
    .await?;
    if report.has_errors() {
        activation.clear()?;
        return Err(format!(
            "plugin activation reported errors: {:?}",
            report.diagnostics
        )
        .into());
    }

    let decision_marks = Arc::new(AtomicUsize::new(0));
    let captured_decision_marks = Arc::clone(&decision_marks);
    register_subscriber(
        EVENT_SUBSCRIBER_NAME,
        Arc::new(move |event| {
            if event.name() == "switchyard.routing.decision"
                && event
                    .data()
                    .and_then(|data| data.get("selected_target"))
                    .and_then(Json::as_str)
                    == Some("embedded")
            {
                captured_decision_marks.fetch_add(1, Ordering::SeqCst);
            }
        }),
    )?;

    let openinference_exporter = InMemorySpanExporterBuilder::new().build();
    let openinference_provider = SdkTracerProvider::builder()
        .with_simple_exporter(openinference_exporter.clone())
        .build();
    let openinference = OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
        openinference_provider,
        "switchyard-embedded-host-e2e",
        OpenTelemetryType::OpenInference,
        OpenTelemetrySubscriberOptions {
            mark_projection: MarkProjection::Tool,
            ..Default::default()
        },
    )?;
    openinference.register(OPENINFERENCE_SUBSCRIBER_NAME)?;

    let agent_scope = push_scope(
        PushScopeParams::builder()
            .name(AGENT_SCOPE_NAME)
            .scope_type(ScopeType::Agent)
            .input(json!({"task": "route an embedded LLM call"}))
            .build(),
    )?;

    let original_provider_called = Arc::new(AtomicBool::new(false));
    let captured_original_provider_called = Arc::clone(&original_provider_called);
    let original_provider: LlmExecutionNextFn = Arc::new(move |_request| {
        captured_original_provider_called.store(true, Ordering::SeqCst);
        Box::pin(async {
            Err(FlowError::Internal(
                "embedded host provider callback must not be called".into(),
            ))
        })
    });

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai.chat_completions")
            .request(LlmRequest {
                headers: Map::from_iter([(
                    "authorization".into(),
                    Json::String(CALLER_CREDENTIAL.into()),
                )]),
                content: json!({
                    "model": "caller/model",
                    "messages": [{"role": "user", "content": "embedded host probe"}],
                    "stream": false
                }),
            })
            .func(original_provider)
            .build(),
    )
    .await;
    let agent_pop_result = pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&agent_scope.uuid)
            .output(json!({"status": "complete"}))
            .build(),
    );

    let flush_result = flush_subscribers();
    let openinference_flush_result = openinference.force_flush();
    let spans_result = openinference_exporter.get_finished_spans();
    let openinference_deregister_result = openinference.deregister(OPENINFERENCE_SUBSCRIBER_NAME);
    let deregister_result = deregister_subscriber(EVENT_SUBSCRIBER_NAME);
    let clear_result = activation.clear();

    flush_result?;
    agent_pop_result?;
    openinference_flush_result?;
    let spans = spans_result?;
    if !openinference_deregister_result? {
        return Err("OpenInference subscriber was not registered during cleanup".into());
    }
    if !deregister_result? {
        return Err("embedded-host subscriber was not registered during cleanup".into());
    }
    clear_result?;
    openinference.shutdown()?;

    let response = result?;
    if response.get("model").and_then(Json::as_str) != Some("fake/embedded") {
        return Err(format!("unexpected routed response: {response}").into());
    }
    if original_provider_called.load(Ordering::SeqCst) {
        return Err("Switchyard called the embedded host's original provider callback".into());
    }
    let decisions = decision_marks.load(Ordering::SeqCst);
    if decisions != 1 {
        return Err(
            format!("expected one Switchyard routing decision mark, observed {decisions}").into(),
        );
    }
    assert_openinference_telemetry(&spans)?;

    println!(
        "{}",
        json!({
            "dynamic_plugin_loaded": true,
            "selected_target": "embedded",
            "response_model": response["model"],
            "original_host_provider_called": false,
            "routing_decision_marks": decisions,
            "openinference_llm_spans": 1,
            "openinference_routing_spans": 1,
            "openinference_parentage_valid": true,
            "openinference_topology": "agent -> [llm, switchyard.routing.decision]",
            "llm_child_parentage_supported": false,
            "credentials_redacted": true
        })
    );
    Ok(())
}
