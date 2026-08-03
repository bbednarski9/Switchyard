# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Switchyard plugin against an embedded NeMo Relay core host."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
CRATE_ROOT = HERE.parents[1]
PACKAGE_SCRIPT = CRATE_ROOT / "scripts" / "package_bundle.py"
FAKE_PROVIDER = HERE.parent / "e2e" / "fake_provider.py"


def free_port() -> int:
    """Reserve an available loopback port for the fake provider."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def get_json(url: str) -> dict[str, Any]:
    """Read one JSON object from the fake provider."""
    with urllib.request.urlopen(url, timeout=5) as response:
        return cast(dict[str, Any], json.loads(response.read()))


def wait_until_ready(provider_url: str, provider: subprocess.Popen[str]) -> None:
    """Wait until the fake provider accepts requests or exits."""
    deadline = time.time() + 10
    while True:
        if provider.poll() is not None:
            output = provider.stdout.read() if provider.stdout is not None else ""
            raise RuntimeError(f"fake provider exited early:\n{output}")
        try:
            if get_json(f"{provider_url}/healthz").get("ok") is True:
                return
        except (OSError, urllib.error.URLError):
            pass
        if time.time() >= deadline:
            raise TimeoutError("fake provider did not become healthy")
        time.sleep(0.05)


def main() -> None:
    """Package the cdylib, start a provider, and execute the embedded host."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-library", required=True, type=Path)
    parser.add_argument("--cargo-target-dir", type=Path)
    args = parser.parse_args()

    library = args.plugin_library.resolve()
    if not library.is_file():
        parser.error(f"compiled plugin library does not exist: {library}")

    temporary = tempfile.TemporaryDirectory(prefix="switchyard-embedded-host-e2e-")
    root = Path(temporary.name)
    bundle = root / "bundle"
    subprocess.run(
        [
            sys.executable,
            str(PACKAGE_SCRIPT),
            "--library",
            str(library),
            "--output",
            str(bundle),
        ],
        check=True,
    )

    provider_port = free_port()
    provider_url = f"http://127.0.0.1:{provider_port}"
    provider = subprocess.Popen(
        [sys.executable, "-u", str(FAKE_PROVIDER), "--port", str(provider_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_until_ready(provider_url, provider)
        environment = os.environ.copy()
        if args.cargo_target_dir is not None:
            environment["CARGO_TARGET_DIR"] = str(args.cargo_target_dir.resolve())
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--locked",
                "--quiet",
                "--manifest-path",
                str(HERE / "Cargo.toml"),
                "--",
                str(bundle / "relay-plugin.toml"),
                provider_url,
            ],
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        calls = get_json(f"{provider_url}/calls")
        if calls.get("fake/embedded") != 1:
            raise AssertionError(
                "expected the Switchyard-owned client to call fake/embedded once; "
                f"provider counters were {calls}"
            )
        print(completed.stdout, end="")
        print(json.dumps({"switchyard_owned_provider_calls": 1}))
    finally:
        provider.send_signal(signal.SIGTERM)
        try:
            provider.wait(timeout=5)
        except subprocess.TimeoutExpired:
            provider.kill()
            provider.wait()
        temporary.cleanup()


if __name__ == "__main__":
    main()
