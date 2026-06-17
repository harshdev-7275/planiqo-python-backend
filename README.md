# ai-service

Standalone AI service for the PM monorepo. Python 3.12, FastAPI, managed with [uv](https://github.com/astral-sh/uv).

> **Status:** scaffold. No AI features wired yet — endpoints and integration come in follow-up changes.

---

## Layout

```
ai-service/
├── src/ai_service/        # application code
│   ├── main.py            # FastAPI app factory + uvicorn entry
│   ├── config.py          # pydantic-settings — single source of truth for env
│   ├── logging.py         # structured JSON logging
│   ├── core/              # cross-cutting (errors, lifespan)
│   └── api/               # routers (health, future /v1/...)
├── tests/                 # pytest, 80%+ coverage gate
├── pyproject.toml         # deps + ruff + mypy + pytest config
├── requirements.txt       # runtime deps (mirror of pyproject for non-uv consumers)
└── requirements-dev.txt   # dev deps
```

## Quickstart

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync deps and create .venv
cd ai-service
uv sync --all-extras

# 3. Configure
cp .env.example .env

# 4. Run
uv run ai-service          # starts uvicorn on $HOST:$PORT (default 0.0.0.0:8000)
# or:
uv run uvicorn ai_service.main:app --reload
```

Visit:
- http://localhost:8000/ — service info
- http://localhost:8000/health — liveness
- http://localhost:8000/health/ready — readiness
- http://localhost:8000/docs — OpenAPI UI (auto-generated)
- http://localhost:8000/redoc — ReDoc UI

## Development

```bash
uv run pytest               # run tests with coverage (80% gate enforced)
uv run pytest --watch       # watch mode (requires pytest-watch)
uv run ruff check .         # lint
uv run ruff format .        # format
uv run mypy .               # strict typecheck
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info (name, version, env) |
| GET | `/health` | Liveness — process is up |
| GET | `/health/ready` | Readiness — ready to take traffic |
| GET | `/docs` | Swagger UI (dev-friendly) |
| GET | `/redoc` | ReDoc UI |

## Adding a new endpoint

1. Add a router in `src/ai_service/api/<name>.py`
2. Wire it in `src/ai_service/api/routes.py`
3. Define Pydantic schemas in `src/ai_service/schemas/<name>.py`
4. Write a failing test in `tests/test_<name>.py`
5. Implement, then refactor

## Deployment

Production runs as a single uvicorn worker behind a reverse proxy (nginx, Caddy, or a cloud load balancer). The service is stateless — scale horizontally by adding instances.

```bash
uv run uvicorn ai_service.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --proxy-headers \
    --forwarded-allow-ips="*"
```

## Environment variables

See `.env.example`. All vars are validated at startup via `pydantic-settings` — a missing or malformed value fails fast with a clear error.
