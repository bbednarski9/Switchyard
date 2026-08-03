<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Embedded Relay host E2E

This standalone harness proves that the packaged Switchyard plugin routes an
LLM call when loaded through NeMo Relay core's public Rust API. It does not
link the plugin as an `rlib`, launch `nemo-relay`, construct CLI gateway state,
or use Relay's provider transport.

Build the plugin and run the proof on macOS with:

```bash
cargo build --release -p switchyard-nemo-relay-plugin
python3 crates/switchyard-nemo-relay-plugin/tests/embedded_host/run.py \
  --plugin-library target/release/libswitchyard_nemo_relay_plugin.dylib
```

Use the corresponding `.so` library on Linux. The runner materializes a plugin
bundle, launches the existing fake provider, and starts a standalone Rust host
that:

1. activates the plugin with `PluginHostActivation::activate`;
2. invokes `llm_call_execute` directly;
3. verifies the selected provider response came from Switchyard's HTTP client;
4. verifies Relay's original provider callback was not invoked;
5. verifies a real `switchyard.routing.decision` mark was emitted;
6. uses Relay's public OpenInference subscriber with an in-memory span exporter
   to verify the managed call produces a completed `LLM` span;
7. projects the decision mark as a `TOOL` span and verifies that it and the
   managed `LLM` span are sibling children of the same exported `AGENT` span;
8. verifies the trace/span IDs, event parent UUIDs, and routing attributes; and
9. verifies neither the caller credential nor the target credential appears in
   exported OpenInference telemetry.

The harness intentionally depends on the published `nemo-relay =
"=0.7.0-rc.4"` package while `0.7.0` is being prepared. The production plugin
dependency and documented compatibility floor target stable Relay `0.7.0`.

## Relay `0.7.0-rc.4` conformance result

The RC successfully loads the plugin, routes the managed HTTP call, and exports
a completed OpenInference `LLM` span with request, response, model, and token
data. The decision span has the expected `random` algorithm and `embedded`
target attributes, and neither configured nor caller credentials are exported.

With the explicit exported Agent scope used by the harness, the managed `LLM`
span and `switchyard.routing.decision` `TOOL` span are non-orphan siblings in
one trace. This is the complete topology supported by the RC.

Native execution callbacks can capture a scope-stack handle, but Relay's public
RC plugin API does not expose the causally active managed-LLM event as a mark
parent. The decision therefore cannot be a child/event of the active LLM span.
A direct `llm_call_execute` invocation without an explicit exported scope would
export the decision as an orphan in a separate trace. The harness reports this
constraint as `"llm_child_parentage_supported": false` rather than implying
that the stronger topology is available.
