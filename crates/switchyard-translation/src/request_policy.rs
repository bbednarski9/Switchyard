// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure target-request policies applied after provider-format translation.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Result, TranslationError};

/// Cache lifetime supported by Anthropic prompt-cache breakpoints.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AnthropicPromptCacheTtl {
    /// Five-minute ephemeral cache entry.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// One-hour ephemeral cache entry.
    #[serde(rename = "1h")]
    OneHour,
}

impl AnthropicPromptCacheTtl {
    /// Returns the Anthropic wire value for this lifetime.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FiveMinutes => "5m",
            Self::OneHour => "1h",
        }
    }
}

/// Adds an Anthropic prompt-cache breakpoint to the final system text block.
///
/// Translated requests normally encode the system prompt as a string. This
/// converts it to Anthropic's equivalent content-block form and attaches the
/// requested cache policy. Existing content-block cache policies are preserved.
/// Returns `true` when a breakpoint was added and `false` when the request has
/// no non-empty system text or already carries a policy on the selected block.
pub fn apply_anthropic_prompt_cache(
    body: &mut Value,
    ttl: AnthropicPromptCacheTtl,
) -> Result<bool> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| TranslationError::InvalidType {
            path: "$".to_string(),
            expected: "object",
        })?;
    let Some(system) = object.get_mut("system") else {
        return Ok(false);
    };
    let cache_control = || json!({"type": "ephemeral", "ttl": ttl.as_str()});

    match system {
        Value::String(text) if !text.is_empty() => {
            *system = json!([{
                "type": "text",
                "text": text,
                "cache_control": cache_control(),
            }]);
            Ok(true)
        }
        Value::String(_) | Value::Null => Ok(false),
        Value::Array(blocks) => {
            let Some(block) = blocks.iter_mut().rev().find_map(|block| {
                let block = block.as_object_mut()?;
                (block.get("type").and_then(Value::as_str) == Some("text")
                    && block
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.is_empty()))
                .then_some(block)
            }) else {
                return Ok(false);
            };
            if block.contains_key("cache_control") {
                return Ok(false);
            }
            block.insert("cache_control".to_string(), cache_control());
            Ok(true)
        }
        _ => Err(TranslationError::InvalidType {
            path: "$.system".to_string(),
            expected: "string or array",
        }),
    }
}
