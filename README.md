# SHL AI Agent — Conversational Assessment Recommendation Engine

A production-ready conversational AI agent that recommends SHL Individual Test Solutions using Retrieval-Augmented Generation (RAG), semantic search, and Google Gemini NLG.

---

## Features

- **Conversational Recommendation** — Natural language chat interface for SHL assessment selection
- **Semantic Search** — pgvector-powered cosine similarity search over assessment embeddings
- **Intent Extraction** — Deterministic parsing of hiring intent (role, skills, experience, categories)
- **Multi-turn Conversation** — Clarification, refinement, comparison, and follow-up support
- **Gemini NLG** — Google Gemini 3.5 Flash for natural language responses (with deterministic fallback)
- **Safety Guardrails** — Off-topic detection, prompt injection detection, polite refusal
- **Stateless API** — Every `/chat` request contains full conversation history; no server-side state
- **Local & Production** — `APP_ENV=local` for WSL PostgreSQL, `APP_ENV=prod` for Neon

---

## Architecture

```
shl-ai-agent/
├── app/
│   ├── api/                  # FastAPI route definitions & dependencies
│   │   ├── routes.py         # POST /chat
│   │   ├── health.py         # GET /health
│   │   ├── admin.py          # Admin endpoints (catalog import, embeddings)
│   │   └── dependencies.py   # FastAPI dependency injection container
│   ├── core/
│   │   ├── config.py         # Pydantic BaseSettings (reads .env)
│   │   ├── logging_.py       # structlog configuration (JSON output)
│   │   └── constants.py      # Enums & app-wide constants
│   ├── database/
│   │   ├── session.py        # Async engine, SessionLocal, Base, get_db
│   │   ├── base.py           # Re-export of Base
│   │   └── models.py         # ORM: Assessment, AssessmentEmbedding
│   ├── models/
│   │   ├── request.py        # ChatRequest, ChatMessage
│   │   ├── response.py       # ChatResponse, Recommendation, HealthResponse
│   │   ├── catalog.py        # CatalogItem, CatalogSearchResult
│   │   └── intent.py         # HiringIntent, JobLevelEnum, SkillArea
│   ├── services/
│   │   ├── chat_service.py       # Orchestrates chat → recommendations
│   │   ├── catalog_service.py    # Catalog ingestion & CRUD
│   │   ├── retrieval_service.py  # Vector similarity search + keyword re-ranking
│   │   ├── embedding_service.py  # Embedding generation pipeline
│   │   ├── embedding_client.py   # Ollama nomic-embed-text client
│   │   ├── intent_extractor.py   # Deterministic hiring intent extraction
│   │   ├── conversation_parser.py# Conversation validation & parsing
│   │   ├── gemini_service.py     # Google Gemini NLG (with fallback)
│   │   ├── prompts.py            # System prompt (separated from code)
│   │   ├── recommendation_mapper.py # Assessment → Recommendation mapping
│   │   ├── clarification_service.py  # Single-question clarification
│   │   ├── comparison_service.py     # Deterministic assessment comparison
│   │   └── document_formatter.py     # Assessment → semantic document
│   ├── repositories/
│   │   ├── catalog_repository.py    # Catalog DB access
│   │   └── embedding_repository.py  # pgvector storage & search
│   └── main.py               # ASGI app factory, middleware, error handlers
├── tests/
│   ├── test_chat.py          # Chat flow tests (30+ tests)
│   └── test_embeddings.py    # Embedding pipeline tests
├── data/
│   └── catalogue.json        # SHL catalog snapshot
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── alembic.ini
└── .env.example
```

### Request → Response Pipeline

```
POST /chat
  │
  ▼
ConversationParser.parse(messages)
  │  Validates roles, order, alternation
  │  Extracts user/assistant messages
  ▼
IntentExtractor.extract(parsed)
  │  Detects role, skills, experience, categories
  │  Detects refinement, comparison, off-topic, injection
  │  Merges constraints across turns
  ▼
Action Router
  │  Priority: Refusal → Comparison → Clarification → Recommendation
  ▼
RetrievalService.retrieve(query)
  │  Embeds query via Ollama nomic-embed-text
  │  pgvector cosine similarity search (3x overfetch)
  │  Keyword + category re-ranking
  │  Deduplication by entity_id
  ▼
RecommendationMapper.map_many(assessments)
  ▼
GeminiService.generate_recommendation_reply(...)
  │  (Falls back to deterministic reply on failure)
  ▼
ChatResponse
  ├─ reply: str
  ├─ recommendations: list[Recommendation]
  └─ end_of_conversation: bool
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension (or Docker)
- Ollama (for embeddings) — `ollama pull nomic-embed-text`
- Google Gemini API key (optional — deterministic fallback works without it)

### 2. Clone & set up

```bash
cd shl-ai-agent
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials and API keys
```

Set `APP_ENV` in `.env`:
- `APP_ENV=local` — uses `POSTGRES_*` credentials (WSL/local PostgreSQL)
- `APP_ENV=prod` — uses `DATABASE_URL` (Neon connection string)

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the Swagger UI.

### 5. Import catalog & generate embeddings

```bash
# Import SHL catalog from local JSON
curl -X POST http://localhost:8000/admin/catalog/import-local

# Generate embeddings for all assessments
curl -X POST http://localhost:8000/admin/embeddings/generate

# Verify
curl -X GET http://localhost:8000/admin/catalog/count
curl -X GET http://localhost:8000/admin/embeddings/count
```

### 6. Test chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer"}]}'
```

### 7. Run tests

```bash
pytest -v
```

---

## API Reference

### `POST /chat`

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "Hiring a Java developer" }
  ]
}
```

**Response (HTTP 200):**
```json
{
  "reply": "These assessments evaluate key Java competencies...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": false
}
```

### Flows

| Flow | Example | Behavior |
|------|---------|----------|
| Clarification | "I need an assessment" | Asks for role/skills |
| Recommendation | "Hiring a Java developer" | Returns relevant assessments |
| Refinement | "Also add personality tests" | Merges constraints, returns updated list |
| Comparison | "Compare OPQ and GSA" | Side-by-side comparison from catalog data |
| Refusal | "What is the salary range?" | Polite refusal, stays in domain |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/catalog/import-local` | Import catalog from `data/catalogue.json` |
| GET | `/admin/catalog/count` | Count assessments in database |
| POST | `/admin/embeddings/generate` | Generate embeddings for all assessments |
| POST | `/admin/embeddings/refresh` | Delete and regenerate all embeddings |
| GET | `/admin/embeddings/count` | Count stored embeddings |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `local` | `local` or `prod` — switches database config |
| `DATABASE_URL` | — | Neon connection string (used when `APP_ENV=prod`) |
| `POSTGRES_*` | — | Local Postgres credentials (used when `APP_ENV=local`) |
| `GOOGLE_API_KEY` | — | Gemini API key (optional; deterministic fallback works) |
| `LLM_MODEL` | `gemini-3.5-flash` | Gemini model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |

---

## Docker

```bash
docker-compose up --build
```

Starts:
- `shl-ai-agent` — FastAPI app on port **8000**
- `shl-ai-agent-db` — PostgreSQL 17 with pgvector on port **5432**

---

## Development

```bash
# Run with hot-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest -v

# Run specific test file
pytest tests/test_chat.py -v
```

---

## License

Proprietary — SHL Research Intern Take-Home Assessment.