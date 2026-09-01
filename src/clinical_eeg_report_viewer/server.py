"""Standard-library, read-only HTTP server for a verified viewer bundle."""

from __future__ import annotations

import base64
import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .bundle import verify_release_bundle


_MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
}

_APP_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)
_REPORT_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
)


def _basic_authorization_valid(
    header: str,
    expected_token_digest: bytes,
) -> bool:
    scheme, separator, encoded = header.partition(" ")
    if not separator or scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    username_ok = hmac.compare_digest(
        hashlib.sha256(username.encode("utf-8")).digest(),
        hashlib.sha256(b"viewer").digest(),
    )
    password_ok = hmac.compare_digest(
        hashlib.sha256(password.encode("utf-8")).digest(), expected_token_digest
    )
    return username_ok and password_ok


class VerifiedBundleServer(ThreadingHTTPServer):
    """HTTP server carrying only a preverified exact path allowlist."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        bundle_root: Path,
        manifest: Mapping[str, Any],
        access_token: str | None,
    ) -> None:
        self.bundle_root = bundle_root
        self.bundle_manifest = dict(manifest)
        self.allowed_paths = frozenset(str(path) for path in manifest["files"])
        self.access_token_digest = (
            hashlib.sha256(access_token.encode("utf-8")).digest()
            if access_token is not None
            else None
        )
        super().__init__(server_address, handler)


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """GET/HEAD-only handler with no filesystem-derived routing."""

    server: VerifiedBundleServer
    protocol_version = "HTTP/1.1"
    server_version = "ClinicalEEGViewer/1"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)

    def _exact_relative(self) -> str | None:
        parsed = urlsplit(self.path)
        try:
            decoded = unquote(parsed.path, errors="strict")
        except UnicodeDecodeError:
            return None
        if "\x00" in decoded or "\\" in decoded:
            return None
        if decoded == "/":
            return "index.html"
        if not decoded.startswith("/"):
            return None
        relative = decoded[1:]
        if not relative or relative.startswith("/"):
            return None
        parts = relative.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            return None
        # Do not normalize: the requested spelling must exactly match a path
        # that was hash-allowlisted when the release bundle was built.
        return relative if relative in self.server.allowed_paths else None

    def _authorized(self) -> bool:
        expected = self.server.access_token_digest
        if expected is None:
            return True
        return _basic_authorization_valid(
            self.headers.get("Authorization", ""), expected
        )

    def _require_authorization(self) -> None:
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header(
            "WWW-Authenticate",
            'Basic realm="Clinical EEG Viewer", charset="UTF-8"',
        )
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_headers(self, relative: str, size: int) -> None:
        suffix = Path(relative).suffix.lower()
        content_type = _MIME_OVERRIDES.get(
            suffix, mimetypes.guess_type(relative)[0] or "application/octet-stream"
        )
        if ";" not in content_type and content_type.startswith("text/"):
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), usb=(), payment=()",
        )
        self.send_header(
            "Content-Security-Policy",
            _REPORT_CSP if relative.startswith("reports/") else _APP_CSP,
        )
        self.end_headers()

    def _serve(self, *, body: bool) -> None:
        if not self._authorized():
            self._require_authorization()
            return
        relative = self._exact_relative()
        if relative is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        path = self.server.bundle_root.joinpath(*relative.split("/"))
        # The full bundle was verified before bind, and routing is exact.  This
        # final check catches replacement/deletion after startup without ever
        # resolving a user-controlled filesystem path.
        if path.is_symlink() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        raw = path.read_bytes()
        expected = self.server.bundle_manifest["files"][relative]
        if len(raw) != expected["size_bytes"] or hashlib.sha256(raw).hexdigest() != expected[
            "sha256"
        ]:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Bundle integrity failure")
            return
        self._send_headers(relative, len(raw))
        if body:
            self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(body=False)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._require_authorization()
            return
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Read-only service")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST


def make_server(
    bundle_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    access_token: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> VerifiedBundleServer:
    """Verify a release bundle completely, then create (but do not run) a server."""

    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if access_token is not None and (
        not 16 <= len(access_token) <= 512
        or any(ord(character) < 0x20 for character in access_token)
    ):
        raise ValueError(
            "access token must contain 16-512 printable characters"
        )
    raw_root = Path(bundle_root)
    manifest = verify_release_bundle(
        raw_root, expected_manifest_sha256=expected_manifest_sha256
    )
    root = raw_root.resolve(strict=True)
    return VerifiedBundleServer(
        (host, port),
        ViewerRequestHandler,
        bundle_root=root,
        manifest=manifest,
        access_token=access_token,
    )


__all__ = ["VerifiedBundleServer", "ViewerRequestHandler", "make_server"]
