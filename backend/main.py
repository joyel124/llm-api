import os
import io
import json
import httpx
import PyPDF2
import docx2txt
from pathlib import Path
from typing import Optional, AsyncIterator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Provider registry
# ──────────────────────────────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "nebius": {
        "name": "Nebius",
        "base_url": "https://api.tokenfactory.nebius.com/v1",
        "api_key": os.getenv("NEBIUS_API_KEY", ""),
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    },
}

DEFAULT_PROVIDER = "nebius"
DEFAULT_MODEL    = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))


def get_provider(provider_id: str) -> dict:
    p = PROVIDERS.get(provider_id)
    if not p:
        raise HTTPException(400, f"Proveedor desconocido: {provider_id!r}")
    if not p["api_key"]:
        raise HTTPException(503, f"API key de {p['name']} no configurada en el servidor")
    return p


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="UPC ABET API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Extracción de texto
# ──────────────────────────────────────────────────────────────

def extract_pdf(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        return text or "[PDF sin texto extraíble — puede ser imagen escaneada]"
    except Exception as e:
        return f"[Error leyendo PDF: {e}]"

def extract_docx(data: bytes) -> str:
    try:
        text = docx2txt.process(io.BytesIO(data))
        return text.strip() or "[DOCX vacío]"
    except Exception as e:
        return f"[Error leyendo DOCX: {e}]"

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml",
    ".toml", ".ini", ".sh", ".sql", ".rst", ".tex",
}

def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(data)
    if ext in (".docx", ".doc"):
        return extract_docx(data)
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return f"[No se pudo leer el archivo: {filename}]"

# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"role inválido: {v!r}")
        return v

class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str]         = None
    provider: str                = DEFAULT_PROVIDER
    stream: bool                 = True
    temperature: float           = 0.7
    max_tokens: int              = 4096
    system_prompt: Optional[str] = None

# ──────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────

def _build_payload(messages: list[dict], model: str, stream: bool,
                   temperature: float, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

async def _stream_provider(payload: dict, base_url: str, api_key: str) -> AsyncIterator[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f"data: [ERROR] {body.decode(errors='replace')}\n\n"
                return
            async for raw_line in resp.aiter_lines():
                if raw_line:
                    yield f"{raw_line}\n\n"

async def _call_provider_sync(payload: dict, base_url: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(r.status_code, detail=detail)
    return r.json()

# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "providers": {k: bool(v["api_key"]) for k, v in PROVIDERS.items()},
    }


@app.get("/api/providers")
def list_providers():
    """Lista los proveedores disponibles y si tienen API key configurada."""
    return [
        {"id": pid, "name": p["name"], "configured": bool(p["api_key"])}
        for pid, p in PROVIDERS.items()
    ]


@app.get("/api/models")
async def list_models(provider: str = DEFAULT_PROVIDER):
    """Lista modelos del proveedor dado, normalizados con campo is_free."""
    p = get_provider(provider)

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{p['base_url']}/models",
            headers={"Authorization": f"Bearer {p['api_key']}"},
        )
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)

    raw = r.json()
    # Both Nebius and OpenRouter return {"data": [...]}
    items = raw.get("data", raw) if isinstance(raw, dict) else raw

    normalized = []
    for m in items:
        model_id   = m.get("id", "")
        model_name = m.get("name") or m.get("id", "")
        if provider == "openrouter":
            pricing = m.get("pricing", {})
            prompt_price = pricing.get("prompt", "1")
            is_free = str(prompt_price) == "0"
        else:
            is_free = False
        normalized.append({"id": model_id, "name": model_name, "is_free": is_free})

    normalized.sort(key=lambda x: x["name"].lower())
    return normalized


@app.post("/api/chat")
async def chat(req: ChatRequest):
    p = get_provider(req.provider)

    provider_messages = []
    if req.system_prompt:
        provider_messages.append({"role": "system", "content": req.system_prompt})
    provider_messages += [m.model_dump() for m in req.messages]

    payload = _build_payload(
        provider_messages, req.model or DEFAULT_MODEL,
        req.stream, req.temperature, req.max_tokens,
    )

    if req.stream:
        return StreamingResponse(
            _stream_provider(payload, p["base_url"], p["api_key"]),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    return await _call_provider_sync(payload, p["base_url"], p["api_key"])


@app.post("/api/chat/with-files")
async def chat_with_files(
    message: str                  = Form(default=""),
    history: str                  = Form(default="[]"),
    model: Optional[str]          = Form(default=None),
    provider: str                 = Form(default=DEFAULT_PROVIDER),
    system_prompt: Optional[str]  = Form(default=None),
    temperature: float            = Form(default=0.7),
    max_tokens: int               = Form(default=4096),
    stream: bool                  = Form(default=True),
    files: list[UploadFile]       = File(default=[]),
):
    p = get_provider(provider)

    # ── Parsear y validar historial ──────────────────────────
    try:
        raw_history = json.loads(history)
        if not isinstance(raw_history, list):
            raw_history = []
    except Exception:
        raw_history = []

    VALID_ROLES = {"user", "assistant", "system"}
    clean_history = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in raw_history
        if isinstance(m, dict) and m.get("role") in VALID_ROLES
    ]

    # ── Extraer texto de archivos ────────────────────────────
    file_blocks: list[str] = []
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                413,
                f"'{f.filename}' ({size_mb:.1f} MB) supera el límite de {MAX_FILE_SIZE_MB} MB",
            )
        extracted = extract_text(f.filename, data)
        file_blocks.append(f"### Archivo: {f.filename}\n\n{extracted}")

    # ── Construir mensaje del usuario ────────────────────────
    parts: list[str] = []
    if message.strip():
        parts.append(message.strip())
    if file_blocks:
        separator = "\n\n---\n\n"
        parts.append("**Archivos adjuntos:**\n\n" + separator.join(file_blocks))

    if not parts:
        raise HTTPException(422, "Debes enviar un mensaje de texto, archivos, o ambos.")

    user_content = "\n\n".join(parts)

    # ── Ensamblar mensajes ───────────────────────────────────
    provider_messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        provider_messages.append({"role": "system", "content": system_prompt.strip()})
    provider_messages.extend(clean_history)
    provider_messages.append({"role": "user", "content": user_content})

    payload = _build_payload(
        provider_messages, model or DEFAULT_MODEL,
        stream, temperature, max_tokens,
    )

    if stream:
        return StreamingResponse(
            _stream_provider(payload, p["base_url"], p["api_key"]),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    return await _call_provider_sync(payload, p["base_url"], p["api_key"])


# ──────────────────────────────────────────────────────────────
# Frontend estático
# ──────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent.parent / "frontend" / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
