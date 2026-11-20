# AURIX AI

FastAPI backend for the AURIX AI chatbot. It verifies Supabase JWTs, stores chat history in Postgres, answers questions about uploaded files, and streams replies from Gemma through the Google Gen AI SDK.

## Requirements

- Python 3.11 or newer
- A Supabase project (Auth JWKS and a session-pooler Postgres URL)
- A Google AI Studio / Gemini API key

## Local setup (Windows)

From the project root:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values. Do not commit `.env`.

| Key | Purpose |
| --- | --- |
| `SUPABASE_URL` | Project URL used to fetch the Auth JWKS |
| `DATABASE_URL` | `postgresql+asyncpg://...` session pooler URL |
| `GOOGLE_API_KEY` | Google Gen AI API key |
| `PRIMARY_MODEL` | Chat model, for example `groq:openai/gpt-oss-120b` or `gemini-3.5-flash-lite` (the older name `GEMMA_MODEL` still works) |
| `FALLBACK_MODEL` | Tried in order after a 429/5xx from the primary; comma-separated, for example `gemini-3.5-flash-lite,gemma-4-26b-a4b-it` |
| `GROQ_API_KEY` | Optional. Needed only for `groq:` model ids (key from console.groq.com) |
| `LIGHT_MODEL` | Short helper calls (suggestions, memory), for example `gemini-3.5-flash-lite` |
| `GEMINI_THINKING` / `GEMMA_THINKING` | Thinking level: Gemini empty (model default) or `low`/`medium`/`high`; Gemma `minimal` (off) or `high` |
| `ALLOWED_ORIGINS` | Comma-separated browser origins |

Passwords that contain characters such as `@`, `#`, or `%` can be stored raw in `DATABASE_URL`. The app percent-encodes the userinfo before opening the pool, drops libpq-only query parameters such as `sslmode` (asyncpg rejects them), and disables asyncpg's prepared-statement cache (`statement_cache_size=0`) for the Supabase pooler. The connection is encrypted the same way as Supabase's `sslmode=require`: TLS is required, and the pooler certificate chain is not verified, which is what lets the session pooler connect from Windows.

Start the API:

```powershell
uvicorn app.main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health check (no auth): http://127.0.0.1:8000/health

Tables are created on startup. Row Level Security is enabled on every table so the Supabase Data API cannot read chat data. The backend connects as the database owner and filters every query by the JWT `sub`.

### Tests

```powershell
pytest
```

The tests use an in-memory SQLite database and a fake model client. They do not call Google or your Supabase database.

## API

All routes except `GET /health` require `Authorization: Bearer <supabase access token>`. The token must use ES256 or RS256, audience `authenticated`, and a valid `sub`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `GET` | `/api/conversations` | Newest conversations, with a last-message preview |
| `POST` | `/api/conversations` | Create an empty chat |
| `GET` | `/api/conversations/{id}` | Messages (sources and feedback) and files |
| `PATCH` | `/api/conversations/{id}` | Rename. Body: `{"title": "..."}` |
| `DELETE` | `/api/conversations/{id}` | Delete a chat |
| `POST` | `/api/conversations/{id}/files` | Multipart field `files` (max 5, 20MB each) |
| `DELETE` | `/api/files/{id}` | Delete one file |
| `POST` | `/api/conversations/{id}/chat` | SSE chat. Body: `{"message": "...", "use_files": true}` |
| `POST` | `/api/messages/{id}/feedback` | Body: `{"value": "up"\|"down", "reason": "optional"}` |
| `DELETE` | `/api/memories` | Delete stored user facts |

Chat responses are Server-Sent Events:

```text
data: {"type":"token","text":"..."}
data: {"type":"done","message_id":"...","sources":[],"suggestions":[]}
```

A failure is `data: {"type":"error","message":"..."}`. The chat route is limited to 30 requests per minute per IP.

Allowed uploads: pdf, docx, txt, md, csv, xlsx, jpg, jpeg, png, webp. Documents and spreadsheets are searched as text chunks (about 800 words with a 100-word overlap) using BM25. CSV and XLSX prompts also include column names and a pandas summary. Images are sent to the model as bytes.

## Deploy on Render

1. Push this repository to the Git host Render will read.
2. Create a **Web Service**. Choose **Docker** so Render builds the included `Dockerfile`. The container starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Add the environment variables from `.env.example`. `ALLOWED_ORIGINS` must include the deployed frontend origin. Render injects `PORT`.
4. Set the health check path to `/health`.
5. Deploy. The service creates its tables on boot.

To deploy without Docker, use a Python 3.11+ native runtime instead:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Copy each file to the same path in your repo (backend/ -> backend repo, frontend/ -> frontend repo).
Delete the stray flat files at the repo roots: chat.py groq.py llm.py schemas.py test_api.py (backend) and
ChatApp.tsx CitationPanel.tsx api.ts api.test.ts greeting.ts greeting.test.ts (frontend).


## Request flow

1. The caller presents a Supabase access token. The API caches the JWKS from `{SUPABASE_URL}/auth/v1/.well-known/jwks.json` and checks the signature, expiry, and audience.
2. A chat message is screened for length and a small set of blocked phrases. Unsafe input is stored and streamed back as a refusal.
3. If the conversation has files and `use_files` is true, the top BM25 chunks (and spreadsheet summaries, plus any images) are added to the prompt, along with saved user memories and the last 10 messages.
4. Gemma is streamed with temperature `0.3`. Thought parts are not forwarded. 429 and 5xx responses are retried up to three times.
5. The user and assistant messages are saved, a new chat is titled from the first six words, three follow-up suggestions are requested, and durable user facts are extracted in the background.
