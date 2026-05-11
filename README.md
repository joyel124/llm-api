# UPC ABET LLM

LLM con modelos open-source de **Nebius Token Factory**.
Soporta archivos adjuntos (PDF, DOCX, TXT…), streaming, historial y es usable como API desde otras apps.

```
llm/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── static/
        └── index.html
```

---

## 1. Instalación

```bash
cd backend
pip install -r requirements.txt
```

## 2. Ejecutar

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Abre en el navegador: **http://localhost:8000**

---

## API — Endpoints

### `GET /api/health`
Verifica que el servidor esté activo y la key configurada.

### `GET /api/models`
Lista todos los modelos disponibles en tu cuenta Nebius.

---

### `POST /api/chat`
Chat JSON puro, sin archivos.

```json
{
  "messages": [
    {"role": "user", "content": "Hola, ¿cómo estás?"}
  ],
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 2048,
  "system_prompt": "Eres un asistente experto en análisis legal."
}
```

Si `stream: true` → responde `text/event-stream` (SSE igual que OpenAI).  
Si `stream: false` → responde JSON completo.

**Ejemplo con curl:**
```bash
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explica RAG en 3 líneas"}],"stream":false}'
```

---

### `POST /api/chat/with-files`
Chat multipart con archivos adjuntos. Los archivos se convierten a texto y se incluyen en el prompt.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `message` | form string | Pregunta o instrucción del usuario |
| `history` | form JSON string | Array de `{role, content}` con mensajes previos |
| `files` | file(s) | PDF, DOCX, TXT, MD, CSV, JSON, PY, JS… |
| `model` | form string | Modelo a usar (opcional) |
| `system_prompt` | form string | Instrucción del sistema (opcional) |
| `stream` | form bool | `true` para SSE, `false` para JSON completo |
| `temperature` | form float | 0.0–2.0 (default 0.7) |
| `max_tokens` | form int | Máximo tokens respuesta (default 4096) |

**Ejemplo con Python (desde otra app):**
```python
import requests

with open("rubrica.pdf", "rb") as r, open("trabajo.docx", "rb") as t:
    resp = requests.post(
        "http://localhost:8000/api/chat/with-files",
        data={
            "message": "Evalúa si el trabajo cumple la rúbrica adjunta",
            "history": "[]",
            "stream": "false",
        },
        files=[
            ("files", ("rubrica.pdf", r, "application/pdf")),
            ("files", ("trabajo.docx", t, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ],
    )
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

**Ejemplo con curl:**
```bash
curl http://localhost:8000/api/chat/with-files \
  -F "message=Resume este PDF" \
  -F "stream=false" \
  -F "files=@documento.pdf"
```

---

## Formatos de archivo soportados

| Formato | Extracción |
|---------|------------|
| `.pdf`  | Texto seleccionable (PyPDF2) |
| `.docx` / `.doc` | Texto completo (docx2txt) |
| `.txt` / `.md` | Directo |
| `.csv` / `.json` / `.xml` / `.html` | Como texto |
| `.py` / `.js` / `.ts` | Como código |

> PDFs escaneados (imagen) no tienen texto extraíble. Usa un OCR primero o convierte a `.txt`.