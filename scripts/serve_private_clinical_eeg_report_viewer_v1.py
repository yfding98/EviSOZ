#!/usr/bin/env python3
"""Serve one verified private EEG viewer release bundle over read-only HTTP."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_report_viewer import make_server, verify_release_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Default is loopback only. Pass 0.0.0.0 explicitly for LAN access.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--expected-manifest-sha256",
        default=None,
        help=(
            "Optional external SHA-256 pin for release_bundle.json. The "
            "CLINICAL_EEG_VIEWER_BUNDLE_SHA256 environment variable is used "
            "when this argument is omitted."
        ),
    )
    parser.add_argument(
        "--allow-unauthenticated-lan",
        action="store_true",
        help=(
            "Explicitly permit a non-loopback bind without an access token. "
            "Not recommended for clinical reference data."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the complete file allowlist and exit without binding a socket.",
    )
    return parser


def _lan_addresses(port: int) -> list[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127."):
                addresses.add(f"http://{address}:{port}/")
    except OSError:
        pass
    return sorted(addresses)


def _loopback_bind(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_manifest_sha256 = args.expected_manifest_sha256 or os.environ.get(
        "CLINICAL_EEG_VIEWER_BUNDLE_SHA256"
    ) or None
    if args.verify_only:
        manifest = verify_release_bundle(
            args.bundle, expected_manifest_sha256=expected_manifest_sha256
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "bundle_id": manifest["bundle_id"],
                    **manifest["counts"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    # The secret is deliberately environment-only: it never appears in CLI
    # arguments, bundle manifests, process output or request logs.
    access_token = os.environ.get("CLINICAL_EEG_VIEWER_TOKEN") or None
    if (
        not _loopback_bind(args.host)
        and access_token is None
        and not args.allow_unauthenticated_lan
    ):
        raise SystemExit(
            "Refusing unauthenticated LAN bind. Set CLINICAL_EEG_VIEWER_TOKEN "
            "(recommended) or explicitly pass --allow-unauthenticated-lan."
        )
    server = make_server(
        args.bundle,
        host=args.host,
        port=args.port,
        access_token=access_token,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    bound_host, bound_port = server.server_address[:2]
    print(f"EEG viewer bundle verified: {server.bundle_manifest['bundle_id']}", flush=True)
    print(
        "Authentication: "
        + ("HTTP Basic enabled (username: viewer)" if access_token else "disabled"),
        flush=True,
    )
    if bound_host == "0.0.0.0":
        addresses = _lan_addresses(bound_port)
        print(
            "LAN URLs: " + (", ".join(addresses) if addresses else f"http://<内网IP>:{bound_port}/"),
            flush=True,
        )
    else:
        print(f"URL: http://{bound_host}:{bound_port}/", flush=True)
    print("Read-only service; press Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
