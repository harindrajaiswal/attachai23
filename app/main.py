"""Main application entry point for Orbit AI (`app.main:app`).

Supports:
1. Production ASGI execution via Uvicorn & FastAPI (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`)
2. Built-in asynchronous/HTTP server & test client (`python3 -m app.main`)
"""

import asyncio
import email.parser
import html
import http.cookies
import json
import mimetypes
import os
import re
import sys
import urllib.parse
import io
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.api.admin import AdminApiHandler
from app.api.public import PublicApiHandler
from app.api.system import SystemApiHandler
from app.core.config import Settings
from app.core.dependencies import ServiceContainer
from app.core.exceptions import OrbitError
from app.middleware.authentication import ADMIN_SESSION_COOKIE, verify_admin_credentials
from app.middleware.cors import build_security_headers, resolve_cors_headers
from app.utils.pii import redact_text

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT_DIR / "app" / "templates"
STATIC_DIR = ROOT_DIR / "static"
PUBLIC_DIR = ROOT_DIR / "public"


def build_project_zip_bytes() -> bytes:
    """Create a complete, clean production zip archive of the Orbit AI repository."""
    buf = io.BytesIO()
    include_roots = ("app", "static", "supabase", "tests", "docs")
    root_files = (
        ".python-version",
        "requirements.txt",
        "render.yaml",
        "Dockerfile",
        "pytest.ini",
        ".env.example",
        ".gitignore",
        "README.md",
    )
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rf in root_files:
            p = ROOT_DIR / rf
            if p.is_file():
                zf.write(p, f"orbit-ai/{rf}")
        demo_p = PUBLIC_DIR / "demo.html"
        if demo_p.is_file():
            zf.write(demo_p, "orbit-ai/public/demo.html")
        for folder in include_roots:
            folder_path = ROOT_DIR / folder
            if not folder_path.is_dir():
                continue
            for file_path in sorted(folder_path.rglob("*")):
                if not file_path.is_file():
                    continue
                if "__pycache__" in file_path.parts or file_path.suffix in (".pyc", ".pyo"):
                    continue
                rel = file_path.relative_to(ROOT_DIR).as_posix()
                zf.write(file_path, f"orbit-ai/{rel}")
    return buf.getvalue()


def render_template_file(template_name: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template from `app/templates/` (uses Jinja2 when installed, or built-in compiler)."""
    try:
        import jinja2  # type: ignore

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
        )
        tmpl = env.get_template(template_name)
        return tmpl.render(**context)
    except ImportError:
        pass

    raw = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")

    # Handle {% extends "base.html" %}
    extends_match = re.search(r'\{%\s*extends\s+["\']([^"\']+)["\']\s*%\}', raw)
    if extends_match:
        parent_name = extends_match.group(1)
        parent_raw = (TEMPLATES_DIR / parent_name).read_text(encoding="utf-8")
        block_match = re.search(
            r"\{%\s*block\s+body\s*%\}(.*?)\{%\s*endblock\s*%\}", raw, re.DOTALL
        )
        body_content = block_match.group(1) if block_match else ""
        raw = re.sub(
            r"\{%\s*block\s+body\s*%\}.*?\{%\s*endblock\s*%\}",
            lambda _m: body_content,
            parent_raw,
            flags=re.DOTALL,
        )

    # Resolve {% include "..." %}
    def replace_include(match: re.Match[str]) -> str:
        inc_path = TEMPLATES_DIR / match.group(1)
        return inc_path.read_text(encoding="utf-8") if inc_path.exists() else ""

    for _ in range(3):
        if "{% include" not in raw:
            break
        raw = re.sub(r'\{%\s*include\s+["\']([^"\']+)["\']\s*%\}', replace_include, raw)

    # Resolve simple {% if var %} ... {% endif %}
    def replace_if(match: re.Match[str]) -> str:
        var_name = match.group(1).strip()
        inner = match.group(2)
        return inner if context.get(var_name) else ""

    raw = re.sub(
        r"\{%\s*if\s+([a-zA-Z0-9_]+)\s*%\}(.*?)\{%\s*endif\s*%\}",
        replace_if,
        raw,
        flags=re.DOTALL,
    )

    # Resolve {{ var }}
    def replace_var(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        val = context.get(key, "")
        return html.escape(str(val if val is not None else ""))

    raw = re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", replace_var, raw)
    return raw


def parse_multipart_form(body: bytes, content_type: str) -> Tuple[Dict[str, str], Dict[str, Tuple[str, bytes, str]]]:
    """Parse multipart/form-data request bodies into fields and file tuples (filename, content, mime)."""
    fields: Dict[str, str] = {}
    files: Dict[str, Tuple[str, bytes, str]] = {}
    if "boundary=" not in content_type:
        return fields, files

    header_bytes = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = email.parser.BytesParser().parsebytes(header_bytes + body)
    if not msg.is_multipart():
        return fields, files

    for part in msg.get_payload():
        disp = part.get("Content-Disposition", "")
        if "form-data" not in disp:
            continue
        name_m = re.search(r'name="([^"]+)"', disp)
        if not name_m:
            continue
        field_name = name_m.group(1)
        filename_m = re.search(r'filename="([^"]*)"', disp)
        payload_bytes = part.get_payload(decode=True) or b""
        if filename_m is not None:
            fname = filename_m.group(1)
            f_mime = part.get_content_type() or "application/octet-stream"
            files[field_name] = (fname, payload_bytes, f_mime)
        else:
            fields[field_name] = payload_bytes.decode("utf-8", errors="replace")
    return fields, files


@dataclass
class HttpResponseData:
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    media_type: str = "application/json"

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class OrbitApplication:
    """Unified Orbit AI application router compatible with FastAPI/ASGI, HTTP Server, and TestClient."""

    def __init__(self, settings: Optional[Settings] = None, container: Optional[ServiceContainer] = None) -> None:
        self.container = container or ServiceContainer(settings=settings)
        self.public_api = PublicApiHandler(self.container)
        self.admin_api = AdminApiHandler(self.container)
        self.system_api = SystemApiHandler(self.container)

    async def ensure_initialized(self) -> None:
        if not self.container._initialized:
            seed = self.container.settings.app_env == "development"
            await self.container.initialize(seed_demo_data=seed)

    def _json_response(
        self,
        data: Any,
        status_code: int = 200,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> HttpResponseData:
        headers = build_security_headers(self.container.settings.is_production)
        if extra_headers:
            headers.update(extra_headers)
        headers["Content-Type"] = "application/json; charset=utf-8"
        return HttpResponseData(
            status_code=status_code,
            headers=headers,
            body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            media_type="application/json",
        )

    def _html_response(
        self,
        html_str: str,
        status_code: int = 200,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> HttpResponseData:
        headers = build_security_headers(self.container.settings.is_production)
        if extra_headers:
            headers.update(extra_headers)
        headers["Content-Type"] = "text/html; charset=utf-8"
        return HttpResponseData(
            status_code=status_code,
            headers=headers,
            body=html_str.encode("utf-8"),
            media_type="text/html",
        )

    async def dispatch(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: bytes = b"",
        cookies: Optional[Dict[str, str]] = None,
        client_ip: str = "127.0.0.1",
    ) -> HttpResponseData:
        """Dispatch an incoming HTTP request to the matching System, Public, Admin, or Static route."""
        await self.ensure_initialized()
        method_u = method.upper()
        clean_path = "/" + path.split("?")[0].lstrip("/")
        if clean_path != "/" and clean_path.endswith("/"):
            clean_path = clean_path.rstrip("/")
        q_params = query_params or {}
        norm_headers = {k.lower(): v for k, v in (headers or {}).items()}
        req_cookies = dict(cookies or {})
        if not req_cookies and "cookie" in norm_headers:
            c_obj = http.cookies.SimpleCookie()
            c_obj.load(norm_headers["cookie"])
            for ck, morsel in c_obj.items():
                req_cookies[ck] = morsel.value

        origin = norm_headers.get("origin") or norm_headers.get("referer")
        is_public_v1 = clean_path.startswith("/v1/") or clean_path.startswith("/static/")
        cors_headers = resolve_cors_headers(
            origin=origin,
            allowed_origins=self.container.settings.allowed_origins,
            is_public_widget_route=is_public_v1,
        )

        if method_u == "OPTIONS":
            return HttpResponseData(status_code=204, headers=cors_headers, body=b"")

        try:
            # --- 1. SYSTEM ENDPOINTS (/health, /ready, /metrics) ---
            if clean_path == "/health" and method_u == "GET":
                code, payload = await self.system_api.get_health()
                return self._json_response(payload, status_code=code, extra_headers=cors_headers)

            if clean_path == "/ready" and method_u == "GET":
                code, payload = await self.system_api.get_ready()
                return self._json_response(payload, status_code=code, extra_headers=cors_headers)

            if clean_path == "/metrics" and method_u == "GET":
                code, payload = await self.system_api.get_metrics(
                    admin_header_key=norm_headers.get("x-admin-api-key"),
                    session_cookie=req_cookies.get(ADMIN_SESSION_COOKIE),
                )
                return self._json_response(payload, status_code=code, extra_headers=cors_headers)

            # --- 2. STATIC ASSETS & PUBLIC DEMO ---
            if clean_path.startswith("/static/") and method_u == "GET":
                rel = clean_path[len("/static/") :]
                target = (STATIC_DIR / rel).resolve()
                if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                    return self._json_response({"error": "Static file not found."}, status_code=404)
                mime, _ = mimetypes.guess_type(str(target))
                resp_headers = build_security_headers(self.container.settings.is_production)
                resp_headers.update(cors_headers)
                resp_headers["Content-Type"] = mime or "application/octet-stream"
                return HttpResponseData(
                    status_code=200,
                    headers=resp_headers,
                    body=target.read_bytes(),
                    media_type=mime or "application/octet-stream",
                )

            if clean_path in ("/orbit-ai-full.zip", "/download/orbit-ai.zip", "/public/orbit-ai-full.zip") and method_u == "GET":
                zip_bytes = build_project_zip_bytes()
                resp_headers = build_security_headers(self.container.settings.is_production)
                resp_headers.update(cors_headers)
                resp_headers["Content-Type"] = "application/zip"
                resp_headers["Content-Disposition"] = 'attachment; filename="orbit-ai-full.zip"'
                return HttpResponseData(
                    status_code=200,
                    headers=resp_headers,
                    body=zip_bytes,
                    media_type="application/zip",
                )

            if clean_path in ("/demo.html", "/public/demo.html") and method_u == "GET":
                demo_file = PUBLIC_DIR / "demo.html"
                return self._html_response(demo_file.read_text(encoding="utf-8"), extra_headers=cors_headers)

            if clean_path == "/api/demo-tenants" and method_u == "GET":
                tenants = await self.container.client_service.list_tenants()
                demo_list = [
                    {
                        "id": t["id"],
                        "business_name": t["business_name"],
                        "slug": t["slug"],
                        "assistant_name": t["assistant_name"],
                        "demo_api_key": self.container.demo_public_keys.get(t["id"], ""),
                    }
                    for t in tenants
                    if t.get("enabled")
                ]
                return self._json_response({"tenants": demo_list}, extra_headers=cors_headers)

            # --- 3. ROOT & ADMIN HTML VIEWS (/, /admin, /admin/login) ---
            if clean_path in ("/", "/admin") and method_u == "GET":
                session_val = req_cookies.get(ADMIN_SESSION_COOKIE)
                csrf_token = self.container.session_manager.verify_session(session_val)
                # Also allow direct header auth or ?autostart=1 in local dev for automated testing
                if not csrf_token and norm_headers.get("x-admin-api-key"):
                    verify_admin_credentials(
                        self.container, admin_header_key=norm_headers.get("x-admin-api-key")
                    )
                    _, csrf_token = self.container.session_manager.create_session()
                if not csrf_token:
                    is_preview = os.environ.get("NODE_ENV") == "development" or not self.container.settings.is_production
                    dev_hint = "orbit-dev-admin-key-2026" if is_preview else ""
                    html_login = render_template_file("login.html", {"dev_hint": dev_hint})
                    return self._html_response(html_login)

                html_dash = render_template_file(
                    "dashboard.html",
                    {
                        "csrf_token": csrf_token,
                        "public_base_url": self.container.settings.public_base_url,
                    },
                )
                return self._html_response(html_dash)

            if clean_path == "/admin/login" and method_u == "GET":
                is_preview = os.environ.get("NODE_ENV") == "development" or not self.container.settings.is_production
                dev_hint = "orbit-dev-admin-key-2026" if is_preview else ""
                html_login = render_template_file("login.html", {"dev_hint": dev_hint})
                return self._html_response(html_login)

            # --- 4. PUBLIC WIDGET API (/v1/*) ---
            if clean_path == "/v1/config" and method_u == "GET":
                cfg = await self.public_api.get_config(
                    api_key=norm_headers.get("x-api-key"),
                    origin=origin,
                    client_id=norm_headers.get("x-client-id") or q_params.get("client_id"),
                    client_ip=client_ip,
                )
                return self._json_response(cfg, extra_headers=cors_headers)

            if clean_path == "/v1/chat" and method_u == "POST":
                json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                res = await self.public_api.post_chat(
                    body=json_body,
                    api_key=norm_headers.get("x-api-key"),
                    origin=origin,
                    client_id=norm_headers.get("x-client-id"),
                    client_ip=client_ip,
                )
                return self._json_response(res, extra_headers=cors_headers)

            if clean_path == "/v1/chat/stream" and method_u == "POST":
                json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                chunks: List[str] = []
                async for sse_chunk in self.public_api.post_chat_stream(
                    body=json_body,
                    api_key=norm_headers.get("x-api-key"),
                    origin=origin,
                    client_id=norm_headers.get("x-client-id"),
                    client_ip=client_ip,
                ):
                    chunks.append(sse_chunk)
                sse_headers = build_security_headers(self.container.settings.is_production)
                sse_headers.update(cors_headers)
                sse_headers.update(
                    {
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    }
                )
                return HttpResponseData(
                    status_code=200,
                    headers=sse_headers,
                    body="".join(chunks).encode("utf-8"),
                    media_type="text/event-stream",
                )

            if clean_path == "/v1/leads" and method_u == "POST":
                json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                status_code, lead_res = await self.public_api.post_leads(
                    body=json_body,
                    api_key=norm_headers.get("x-api-key"),
                    origin=origin,
                    client_id=norm_headers.get("x-client-id"),
                    idempotency_key=norm_headers.get("x-idempotency-key"),
                    client_ip=client_ip,
                )
                return self._json_response(lead_res, status_code=status_code, extra_headers=cors_headers)

            # --- 5. ADMIN API (/api/admin/*) ---
            if clean_path == "/api/admin/login" and method_u == "POST":
                self.container.rate_limiter.check(
                    f"admin_login:{client_ip}",
                    self.container.settings.admin_rate_limit_per_minute,
                )
                json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                raw_key = str(json_body.get("admin_api_key") or norm_headers.get("x-admin-api-key") or "")
                signed_cookie, csrf_token = await self.admin_api.login(raw_key)
                secure_flag = "; Secure" if self.container.settings.is_production else ""
                cookie_header = (
                    f"{ADMIN_SESSION_COOKIE}={signed_cookie}; HttpOnly; Path=/; SameSite=Lax; Max-Age=28800{secure_flag}"
                )
                return self._json_response(
                    {"authenticated": True, "csrfToken": csrf_token},
                    extra_headers={"Set-Cookie": cookie_header, **cors_headers},
                )

            if clean_path == "/api/admin/logout" and method_u == "POST":
                await self.admin_api.logout(req_cookies.get(ADMIN_SESSION_COOKIE))
                clear_cookie = f"{ADMIN_SESSION_COOKIE}=; HttpOnly; Path=/; SameSite=Lax; Max-Age=0"
                return self._json_response(
                    {"authenticated": False},
                    extra_headers={"Set-Cookie": clear_cookie, **cors_headers},
                )

            if clean_path.startswith("/api/admin/"):
                self.container.rate_limiter.check(
                    f"admin_ip:{client_ip}",
                    self.container.settings.admin_rate_limit_per_minute,
                )
                verify_admin_credentials(
                    container=self.container,
                    admin_header_key=norm_headers.get("x-admin-api-key"),
                    session_cookie=req_cookies.get(ADMIN_SESSION_COOKIE),
                    csrf_header=norm_headers.get("x-csrf-token"),
                    method=method_u,
                )

                sub = clean_path[len("/api/admin") :]

                if sub == "/overview" and method_u == "GET":
                    res = await self.admin_api.get_overview(
                        tenant_id=q_params.get("tenant_id"),
                        start_date=q_params.get("start_date"),
                        end_date=q_params.get("end_date"),
                    )
                    return self._json_response(res)

                if sub == "/clients" and method_u == "GET":
                    return self._json_response(await self.admin_api.list_tenants())

                if sub == "/clients" and method_u == "POST":
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    return self._json_response(
                        await self.admin_api.create_tenant(json_body), status_code=201
                    )

                # /clients/{tenant_id}
                m_client = re.match(r"^/clients/([^/]+)$", sub)
                if m_client:
                    tid = m_client.group(1)
                    if method_u == "GET":
                        return self._json_response(await self.admin_api.get_tenant(tid))
                    if method_u in ("PATCH", "PUT"):
                        json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                        return self._json_response(await self.admin_api.update_tenant(tid, json_body))
                    if method_u == "DELETE":
                        return self._json_response(await self.admin_api.delete_tenant(tid))

                # /clients/{tenant_id}/duplicate
                m_dup = re.match(r"^/clients/([^/]+)/duplicate$", sub)
                if m_dup and method_u == "POST":
                    tid = m_dup.group(1)
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    return self._json_response(
                        await self.admin_api.duplicate_tenant(tid, json_body), status_code=201
                    )

                # /clients/{tenant_id}/keys
                m_keys = re.match(r"^/clients/([^/]+)/keys$", sub)
                if m_keys and method_u == "POST":
                    tid = m_keys.group(1)
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    return self._json_response(
                        await self.admin_api.generate_key(tid, name=json_body.get("name", "Widget Key")),
                        status_code=201,
                    )

                # /clients/{tenant_id}/keys/rotate
                m_rot = re.match(r"^/clients/([^/]+)/keys/rotate$", sub)
                if m_rot and method_u == "POST":
                    return self._json_response(await self.admin_api.rotate_key(m_rot.group(1)))

                # /clients/{tenant_id}/keys/{key_id}
                m_kstat = re.match(r"^/clients/([^/]+)/keys/([^/]+)$", sub)
                if m_kstat and method_u == "PATCH":
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    return self._json_response(
                        await self.admin_api.toggle_key(
                            m_kstat.group(1), m_kstat.group(2), bool(json_body.get("enabled", True))
                        )
                    )

                # /clients/{tenant_id}/domains
                m_dom = re.match(r"^/clients/([^/]+)/domains$", sub)
                if m_dom:
                    tid = m_dom.group(1)
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    dom_val = str(json_body.get("domain") or q_params.get("domain") or "")
                    if method_u == "POST":
                        return self._json_response(await self.admin_api.add_domain(tid, dom_val))
                    if method_u == "DELETE":
                        return self._json_response(await self.admin_api.remove_domain(tid, dom_val))

                # Prompts: /prompts/{tenant_id}
                m_pr = re.match(r"^/prompts/([^/]+)$", sub)
                if m_pr:
                    tid = m_pr.group(1)
                    if method_u == "GET":
                        return self._json_response(await self.admin_api.list_prompts(tid))
                    if method_u == "POST":
                        json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                        return self._json_response(await self.admin_api.save_prompt(tid, json_body), status_code=201)

                m_pr_res = re.match(r"^/prompts/([^/]+)/restore/(\d+)$", sub)
                if m_pr_res and method_u == "POST":
                    return self._json_response(
                        await self.admin_api.restore_prompt(m_pr_res.group(1), int(m_pr_res.group(2)))
                    )

                m_pr_prev = re.match(r"^/prompts/([^/]+)/preview$", sub)
                if m_pr_prev and method_u == "POST":
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    return self._json_response(await self.admin_api.preview_prompt(m_pr_prev.group(1), json_body))

                # Knowledge: /knowledge
                if sub == "/knowledge" and method_u == "GET":
                    return self._json_response(await self.admin_api.list_knowledge(q_params.get("tenant_id")))

                if sub == "/knowledge/upload" and method_u == "POST":
                    ctype = norm_headers.get("content-type", "")
                    if "multipart/form-data" in ctype:
                        fields, files = parse_multipart_form(body, ctype)
                        tid = fields.get("tenant_id") or q_params.get("tenant_id") or ""
                        file_tuple = files.get("file")
                        if not file_tuple:
                            raise OrbitError("Missing 'file' in multipart upload.", status_code=422)
                        fname, fbytes, fmime = file_tuple
                        res = await self.admin_api.upload_knowledge(tid, fname, fbytes, fmime)
                        return self._json_response(res, status_code=201)
                    else:
                        json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                        tid = str(json_body.get("tenant_id") or "")
                        fname = str(json_body.get("filename") or "document.txt")
                        fcontent = str(json_body.get("content") or "").encode("utf-8")
                        fmime = str(json_body.get("content_type") or "text/plain")
                        res = await self.admin_api.upload_knowledge(tid, fname, fcontent, fmime)
                        return self._json_response(res, status_code=201)

                m_k_reb = re.match(r"^/knowledge/([^/]+)/([^/]+)/rebuild$", sub)
                if m_k_reb and method_u == "POST":
                    return self._json_response(
                        await self.admin_api.rebuild_knowledge(m_k_reb.group(1), m_k_reb.group(2))
                    )

                m_k_del = re.match(r"^/knowledge/([^/]+)/([^/]+)$", sub)
                if m_k_del and method_u == "DELETE":
                    return self._json_response(
                        await self.admin_api.delete_knowledge(m_k_del.group(1), m_k_del.group(2))
                    )

                # Conversations
                if sub == "/conversations" and method_u == "GET":
                    return self._json_response(
                        await self.admin_api.list_conversations(
                            tenant_id=q_params.get("tenant_id"),
                            search=q_params.get("search", ""),
                        )
                    )

                m_conv = re.match(r"^/conversations/([^/]+)/([^/]+)$", sub)
                if m_conv and method_u == "GET":
                    return self._json_response(
                        await self.admin_api.get_conversation(m_conv.group(1), m_conv.group(2))
                    )

                # Leads
                if sub == "/leads" and method_u == "GET":
                    return self._json_response(
                        await self.admin_api.list_leads(
                            tenant_id=q_params.get("tenant_id"),
                            status=q_params.get("status"),
                            search=q_params.get("search", ""),
                        )
                    )

                if sub == "/leads/export" and method_u == "GET":
                    csv_str = await self.admin_api.export_leads_csv(
                        tenant_id=q_params.get("tenant_id"),
                        status=q_params.get("status"),
                        search=q_params.get("search", ""),
                    )
                    csv_headers = build_security_headers(self.container.settings.is_production)
                    csv_headers["Content-Type"] = "text/csv; charset=utf-8"
                    csv_headers["Content-Disposition"] = 'attachment; filename="orbit_leads_export.csv"'
                    return HttpResponseData(
                        status_code=200,
                        headers=csv_headers,
                        body=csv_str.encode("utf-8"),
                        media_type="text/csv",
                    )

                m_lead_up = re.match(r"^/leads/([^/]+)/([^/]+)$", sub)
                if m_lead_up and method_u == "PATCH":
                    json_body = json.loads(body.decode("utf-8") or "{}") if body else {}
                    return self._json_response(
                        await self.admin_api.update_lead(m_lead_up.group(1), m_lead_up.group(2), json_body)
                    )

                # Notifications & Provider Tests
                if sub == "/notifications" and method_u == "GET":
                    return self._json_response(
                        await self.admin_api.list_notifications(
                            tenant_id=q_params.get("tenant_id"),
                            status=q_params.get("status"),
                        )
                    )

                if sub == "/notifications/retry" and method_u == "POST":
                    return self._json_response(
                        await self.admin_api.retry_notifications(tenant_id=q_params.get("tenant_id"))
                    )

                if sub == "/providers/test" and method_u == "POST":
                    return self._json_response(await self.admin_api.test_providers())

            return self._json_response(
                {"error": f"Route {method_u} {clean_path} not found.", "code": "NOT_FOUND"},
                status_code=404,
                extra_headers=cors_headers,
            )

        except OrbitError as oe:
            safe_msg = redact_text(oe.message, extra_secrets=self.container.settings.known_secrets())
            extra = dict(cors_headers)
            if oe.status_code == 429 and "retry_after" in oe.details:
                extra["Retry-After"] = str(oe.details["retry_after"])
            return self._json_response(
                {"error": safe_msg, "code": oe.code, "details": oe.details},
                status_code=oe.status_code,
                extra_headers=extra,
            )
        except json.JSONDecodeError:
            return self._json_response(
                {"error": "Malformed JSON request payload.", "code": "INVALID_JSON"},
                status_code=400,
                extra_headers=cors_headers,
            )
        except Exception as exc:
            safe_err = redact_text(str(exc), extra_secrets=self.container.settings.known_secrets())
            self.container.logger.error(f"Unhandled error on {method_u} {clean_path}: {safe_err}")
            return self._json_response(
                {"error": "An internal server error occurred.", "code": "INTERNAL_ERROR"},
                status_code=500,
                extra_headers=cors_headers,
            )


def create_app(settings: Optional[Settings] = None) -> Any:
    """Create the Orbit AI application (returns FastAPI ASGI app if installed, else OrbitApplication)."""
    orbit_app = OrbitApplication(settings=settings)
    try:
        from fastapi import FastAPI, Request, Response  # type: ignore
        from app.lifespan import app_lifespan

        fastapi_app = FastAPI(
            title="Orbit AI — Multi-Tenant RAG SaaS",
            version="1.0.0",
            lifespan=app_lifespan,
        )
        fastapi_app.state.container = orbit_app.container
        fastapi_app.state.orbit_app = orbit_app

        @fastapi_app.api_route(
            "/{full_path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def catch_all(request: Request, full_path: str) -> Response:
            body_bytes = await request.body()
            q_params = dict(request.query_params)
            req_headers = dict(request.headers)
            client_ip = request.client.host if request.client else "127.0.0.1"
            resp_data = await orbit_app.dispatch(
                method=request.method,
                path="/" + full_path,
                query_params=q_params,
                headers=req_headers,
                body=body_bytes,
                cookies=dict(request.cookies),
                client_ip=client_ip,
            )
            return Response(
                content=resp_data.body,
                status_code=resp_data.status_code,
                headers=resp_data.headers,
                media_type=resp_data.media_type,
            )

        return fastapi_app
    except ImportError:
        return orbit_app


class OrbitTestClient:
    """Synchronous/Asynchronous test client for OrbitApplication or FastAPI app."""

    def __init__(self, app_instance: Any, client_ip: str = "127.0.0.1") -> None:
        self.orbit_app: OrbitApplication = getattr(app_instance, "state", None) and getattr(
            app_instance.state, "orbit_app", app_instance
        ) or app_instance
        self.client_ip = client_ip
        self.cookies: Dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        json_body: Optional[Any] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        client_ip: Optional[str] = None,
    ) -> HttpResponseData:
        parsed = urllib.parse.urlparse(path)
        q_params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        req_headers = dict(headers or {})
        raw_body = data or b""
        if json_body is not None:
            raw_body = json.dumps(json_body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        resp = asyncio.run(
            self.orbit_app.dispatch(
                method=method,
                path=parsed.path,
                query_params=q_params,
                headers=req_headers,
                body=raw_body,
                cookies=self.cookies,
                client_ip=client_ip or self.client_ip,
            )
        )
        set_cookie = resp.headers.get("Set-Cookie")
        if set_cookie:
            c = http.cookies.SimpleCookie()
            c.load(set_cookie)
            for k, morsel in c.items():
                if morsel.value:
                    self.cookies[k] = morsel.value
                else:
                    self.cookies.pop(k, None)
        return resp

    def get(self, path: str, headers: Optional[Dict[str, str]] = None) -> HttpResponseData:
        return self.request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        json_body: Optional[Any] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HttpResponseData:
        return self.request("POST", path, json_body=json_body, data=data, headers=headers)

    def patch(
        self,
        path: str,
        json_body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HttpResponseData:
        return self.request("PATCH", path, json_body=json_body, headers=headers)

    def delete(self, path: str, headers: Optional[Dict[str, str]] = None) -> HttpResponseData:
        return self.request("DELETE", path, headers=headers)


# Canonical ASGI application object for `uvicorn app.main:app`
app = create_app()


def run_builtin_http_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OrbitApplication over Python's ThreadingHTTPServer when Uvicorn is unavailable."""
    import threading

    orbit_app: OrbitApplication = getattr(app, "state", None) and getattr(
        app.state, "orbit_app", app
    ) or app
    asyncio.run(orbit_app.ensure_initialized())
    dispatch_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            try:
                parsed = urllib.parse.urlparse(self.path)
                q_params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
                length = int(self.headers.get("Content-Length", "0") or "0")
                body_bytes = self.rfile.read(length) if length > 0 else b""
                req_headers = {k: v for k, v in self.headers.items()}
                client_ip = self.client_address[0] if self.client_address else "127.0.0.1"

                with dispatch_lock:
                    resp: HttpResponseData = asyncio.run(
                        orbit_app.dispatch(
                            method=self.command,
                            path=parsed.path,
                            query_params=q_params,
                            headers=req_headers,
                            body=body_bytes,
                            client_ip=client_ip,
                        )
                    )
                self.send_response(resp.status_code)
                for hk, hv in resp.headers.items():
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(resp.body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(resp.body)
            except BrokenPipeError:
                return

        def do_GET(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

        def do_PUT(self) -> None:
            self._handle()

        def do_PATCH(self) -> None:
            self._handle()

        def do_DELETE(self) -> None:
            self._handle()

        def do_OPTIONS(self) -> None:
            self._handle()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Orbit AI Python Server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "8000"))
    host_addr = os.environ.get("HOST", "0.0.0.0")
    for idx, arg in enumerate(sys.argv):
        if arg == "--port" and idx + 1 < len(sys.argv):
            port_num = int(sys.argv[idx + 1])
        elif arg == "--host" and idx + 1 < len(sys.argv):
            host_addr = sys.argv[idx + 1]
    run_builtin_http_server(host=host_addr, port=port_num)
