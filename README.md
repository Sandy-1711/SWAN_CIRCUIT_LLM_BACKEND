# SWAN Circuit LLM Backend

Backend service for SWAN that combines authentication, chat/session management, and LLM-powered IoT circuit generation.

## Project Description

This project exposes a FastAPI backend that:
- manages users and auth (JWT + cookie-based sessions),
- stores chats/messages in PostgreSQL,
- streams model responses for circuit-generation conversations,
- supports multiple inference modes (`baseline`, `chained`, `rag`),
- ingests user feedback into a retrieval store for future RAG responses.

The generation flow is focused on converting natural-language prompts into:
1. Arduino-style source code,
2. intermediate circuit representation (IR),
3. Wokwi-compatible JSON output (`parts` + `connections`).

## Tech Stack

- **Language:** Python
- **API Framework:** FastAPI, Uvicorn
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy
- **Auth/Security:** passlib (bcrypt), python-jose (JWT)
- **LLM Runtime:** llama-cpp-python (local GGUF models)
- **Retrieval/RAG:** sentence-transformers + FAISS
- **ML/LLM ecosystem:** torch, transformers, langchain, langgraph
- **Containerization (DB):** Docker Compose

## Architecture Choices

- **Layered backend structure**
  - `routers/` handles HTTP endpoints and request/response flow.
  - `services/` orchestrates higher-level business logic.
  - `crud.py` encapsulates DB operations.
  - `model.py` and `schemas.py` define persistence and API contracts.

- **Streaming-first inference APIs**
  - Chat routes return `text/event-stream`-style chunked JSON so the frontend can render progressive model output.

- **Multi-strategy generation**
  - `baseline`: single-pass code + JSON generation.
  - `chained`: staged pipeline (code -> IR -> JSON).
  - `rag`: retrieval-augmented variant with similarity threshold abort behavior for low-confidence queries.

- **Feedback loop for retrieval quality**
  - `/user/feedback` ingests edited prompt/code/output back into embedding storage so the retrieval corpus can evolve.

## Repository Structure

- `/app.py` - FastAPI app bootstrap and router registration
- `/routers` - auth/user/chat HTTP routes
- `/services` - inference/chat orchestration
- `/inferences` - model prompting + staged inference logic
- `/pipelines` - RAG pipeline composition
- `/rag` - embedding search and ingestion utilities
- `/models` - local llama-cpp model loader configuration
- `/database.py`, `/model.py`, `/crud.py` - DB config, entities, data access
- `/data/dataset.json` - retrieval dataset seed
- `/docker-compose.yml` - PostgreSQL service for local development

## Prerequisites

- Python 3.10+ (recommended)
- Docker + Docker Compose
- Local model files expected by `models/shared_llms.py`:
  - `models/coder_F32/unsloth.F32.gguf`
  - `models/compressor_F32/unsloth.F32.gguf`
  - `models/generator_F32/unsloth.F32.gguf`
  - `models/baseline_F32/unsloth.F32.gguf`

## Environment Variables

Create a `.env` file from `.env.example` and set:

- `DB_URL` - SQLAlchemy connection string (example: `postgresql://<user>:<password>@<host>:<port>/<database>`)
- `SECRET_KEY` - JWT signing key
- `ALGORITHM` - JWT algorithm (example: `HS256`)
- `ACCESS_TOKEN_EXPIRE_MINUTES` - token expiration in minutes
- `HF_TOKEN` - optional Hugging Face token (if required by your runtime)
- `LLAMA_LOG_LEVEL` - llama runtime logging level

The `DB_URL` example above uses placeholders only. Do not hardcode real credentials in code or docs. `.env` is already gitignored in this repository for local development; for production, prefer secret managers and managed authentication flows over static passwords.

## Setup and Run

1. **Clone and enter project**
   ```bash
   git clone <repo-url>
   cd SWAN_CIRCUIT_LLM_BACKEND
   ```

2. **Start PostgreSQL**
   ```bash
   docker compose up -d
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # then edit .env values
   ```

5. **Run API server**
   ```bash
   python app.py
   ```
   or
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

6. **Health check**
   ```bash
   curl http://localhost:8000/health
   ```

## API Overview

### Root/Health
- `GET /` - basic hello-world response
- `GET /health` - application health

### Auth (`/auth`)
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/health`

### User + Chat (`/user`)
- `GET /user/me`
- `PUT /user/me`
- `GET /user/chat?id=<chat_id>`
- `GET /user/all_chats`
- `POST /user/new_chat?model=<chained|baseline|rag>` (streaming)
- `POST /user/chat_stream?model=<chained|baseline|rag>` (streaming)
- `PUT /user/feedback`

## Notes

- CORS is currently configured for `http://localhost:3000` and `http://localhost:3001`.
- Cookies are set with `domain=localhost` and `secure=False` for local development. In production, enable HTTPS, keep `httponly=True` and set `secure=True` to reduce XSS exposure, and enforce an appropriate `SameSite` value (for example `Lax`/`Strict`) to reduce CSRF risk.
- The project currently has no automated test suite committed in this repository.
