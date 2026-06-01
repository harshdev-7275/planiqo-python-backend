from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from agents import supervisor
from clients.neo4j_client import neo4j_client
from clients.node_api import node_api_client
from config.settings import settings
from middleware.auth import InternalAuthMiddleware


class ChatRequest(BaseModel):
    message: str
    user_id: str
    org_slug: str
    project_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI service starting up")
    neo4j_client.connect()
    yield
    await neo4j_client.close()
    await node_api_client.close()
    logger.info("AI service shut down")


app = FastAPI(title="AI Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(InternalAuthMiddleware)


@app.get("/health")
async def health():
    node_ok = await node_api_client.ping()
    neo4j_ok = await neo4j_client.ping()
    return {
        "status": "ok",
        "node_api": node_ok,
        "neo4j": neo4j_ok,
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    return await supervisor.run(
        message=body.message,
        user_id=body.user_id,
        org_slug=body.org_slug,
        project_id=body.project_id,
    )
