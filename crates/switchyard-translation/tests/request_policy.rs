// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use switchyard_translation::{
    apply_anthropic_prompt_cache, AnthropicPromptCacheTtl, TranslationError,
};

#[test]
fn prompt_cache_converts_system_text_to_anthropic_content_block() {
    let mut body = json!({
        "model": "claude-sonnet",
        "system": "stable instructions",
        "messages": [{"role": "user", "content": "hello"}],
    });

    assert!(
        apply_anthropic_prompt_cache(&mut body, AnthropicPromptCacheTtl::FiveMinutes)
            .expect("policy should apply")
    );
    assert_eq!(
        body["system"],
        json!([{
            "type": "text",
            "text": "stable instructions",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }])
    );
}

#[test]
fn prompt_cache_updates_final_system_text_block_without_overwriting_existing_policy() {
    let mut body = json!({
        "system": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"}
        ]
    });
    assert!(
        apply_anthropic_prompt_cache(&mut body, AnthropicPromptCacheTtl::OneHour)
            .expect("policy should apply")
    );
    assert_eq!(
        body["system"][1]["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );

    assert!(
        !apply_anthropic_prompt_cache(&mut body, AnthropicPromptCacheTtl::FiveMinutes)
            .expect("existing policy should be preserved")
    );
    assert_eq!(body["system"][1]["cache_control"]["ttl"], "1h");
}

#[test]
fn prompt_cache_is_a_noop_without_cacheable_system_text() {
    for mut body in [
        json!({"messages": []}),
        json!({"system": ""}),
        json!({"system": [{"type": "image", "source": {}}]}),
    ] {
        let original = body.clone();
        assert!(
            !apply_anthropic_prompt_cache(&mut body, AnthropicPromptCacheTtl::FiveMinutes)
                .expect("policy should be a no-op")
        );
        assert_eq!(body, original);
    }
}

#[test]
fn prompt_cache_rejects_malformed_request_shapes() {
    let error = apply_anthropic_prompt_cache(&mut json!([]), AnthropicPromptCacheTtl::FiveMinutes)
        .expect_err("non-object body should fail");
    assert!(matches!(error, TranslationError::InvalidType { .. }));

    let error = apply_anthropic_prompt_cache(
        &mut json!({"system": 7}),
        AnthropicPromptCacheTtl::FiveMinutes,
    )
    .expect_err("invalid system value should fail");
    assert!(matches!(error, TranslationError::InvalidType { .. }));
}
