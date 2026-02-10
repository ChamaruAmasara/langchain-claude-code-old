"""
OAuth2 authentication for Anthropic Claude (Pro/Max subscription).

Uses the same OAuth client as Claude Code CLI to obtain Bearer tokens
for the Anthropic Messages API — no API key required.
"""

from __future__ import annotations

import json
import secrets
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from time import time
from typing import Optional

import httpx

# Claude Code's registered OAuth client_id (public)
DEFAULT_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://console.anthropic.com/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8912/callback"
DEFAULT_TOKEN_FILE = Path.home() / ".langchain-claude-code" / "auth.json"


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth redirect callback."""

    auth_code: str | None = None

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>&#9989; Authenticated! You can close this tab.</h2></body></html>"
            )
        else:
            error = params.get("error", ["unknown"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h2>Error: {error}</h2></body></html>".encode())

    def log_message(self, format: str, *args: object) -> None:
        pass


class ClaudeOAuthManager:
    """Manages OAuth2 tokens for Anthropic Claude API.

    Handles the full lifecycle: browser-based authorization, token exchange,
    caching, and automatic refresh.

    Args:
        client_id: OAuth client ID. Defaults to Claude Code's public client ID.
        redirect_uri: Local redirect URI for the OAuth callback.
        token_file: Path to cache tokens on disk. Set to None to disable caching.
        redirect_port: Port for the local OAuth callback server.
    """

    def __init__(
        self,
        client_id: str = DEFAULT_CLIENT_ID,
        redirect_uri: str = DEFAULT_REDIRECT_URI,
        token_file: Optional[Path | str] = DEFAULT_TOKEN_FILE,
        redirect_port: int = 8912,
    ) -> None:
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.token_file = Path(token_file) if token_file else None
        self.redirect_port = redirect_port
        self._tokens: dict | None = None

    def get_access_token(self) -> str:
        """Get a valid access token, refreshing or re-authenticating as needed."""
        if self._tokens is None:
            self._tokens = self._load_tokens()

        if self._tokens:
            now_ms = int(time() * 1000)
            if self._tokens["expires_at"] - now_ms >= 300_000:
                return self._tokens["access_token"]
            # Try refresh
            try:
                self._tokens = self._refresh(self._tokens["refresh_token"])
                self._save_tokens()
                return self._tokens["access_token"]
            except Exception:
                pass

        # Full auth flow
        self._tokens = self._authorize_flow()
        self._save_tokens()
        return self._tokens["access_token"]

    def _authorize_flow(self) -> dict:
        """Run the full browser-based OAuth2 authorization flow."""
        _OAuthCallbackHandler.auth_code = None
        code_verifier = secrets.token_urlsafe(64)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "user:inference",
            "state": secrets.token_urlsafe(32),
            "code_challenge": code_verifier,
            "code_challenge_method": "plain",
        }
        auth_url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

        server = HTTPServer(("127.0.0.1", self.redirect_port), _OAuthCallbackHandler)
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        webbrowser.open(auth_url)
        server_thread.join(timeout=120)
        server.server_close()

        if not _OAuthCallbackHandler.auth_code:
            raise RuntimeError("OAuth flow timed out or was cancelled.")

        return self._exchange_code(_OAuthCallbackHandler.auth_code, code_verifier)

    def _exchange_code(self, auth_code: str, code_verifier: str) -> dict:
        """Exchange authorization code for tokens."""
        resp = httpx.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "code_verifier": code_verifier,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_at": int(time() * 1000) + data.get("expires_in", 3600) * 1000,
        }

    def _refresh(self, refresh_token: str) -> dict:
        """Refresh an expired access token."""
        resp = httpx.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_at": int(time() * 1000) + data.get("expires_in", 3600) * 1000,
        }

    def _load_tokens(self) -> dict | None:
        if self.token_file and self.token_file.exists():
            return json.loads(self.token_file.read_text())
        return None

    def _save_tokens(self) -> None:
        if self.token_file and self._tokens:
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(json.dumps(self._tokens, indent=2))
