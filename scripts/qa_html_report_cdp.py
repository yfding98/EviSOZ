#!/usr/bin/env python3
"""Render a local HTML report in emulated light/dark media using Chrome CDP."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import requests
import websocket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("--scheme", choices=("light", "dark"), default="light")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9222)
    args = parser.parse_args()

    target = requests.put(f"http://127.0.0.1:{args.port}/json/new?about:blank", timeout=5).json()
    ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=10)
    request_id = 0

    def call(method: str, params=None):
        nonlocal request_id
        request_id += 1
        ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(ws.recv())
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    call("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 7200, "deviceScaleFactor": 1, "mobile": False})
    call("Emulation.setEmulatedMedia", {"media": "screen", "features": [{"name": "prefers-color-scheme", "value": args.scheme}]})
    call("Page.navigate", {"url": args.html.resolve().as_uri()})
    time.sleep(3)
    diagnostics = call(
        "Runtime.evaluate",
        {
            "expression": "JSON.stringify({scheme:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light',background:getComputedStyle(document.body).backgroundColor,charts:[...document.querySelectorAll('[data-recharts-chart]')].map(x=>({ready:x.dataset.rechartsReady||'',svg:x.querySelectorAll('[data-recharts-live] svg').length})),overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth})",
            "returnByValue": True,
        },
    )
    screenshot = call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True, "fromSurface": True})
    args.output.write_bytes(base64.b64decode(screenshot["data"]))
    print(json.loads(diagnostics["result"]["value"]))
    ws.close()


if __name__ == "__main__":
    main()
