from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import account, auth, coordinates, inventory, pipeline
from config import root_config
from config.logging_config import setup_logging
from config.models import OpenAIModelGroupConfig
from config.openai_config import OpenAIConfig
from db.demo_data import ensure_demo_inventory_loaded
from db.runtime_init import ensure_runtime_schema

app = FastAPI(
    title="TKNT Pipeline API",
    version="0.2.0",
    description=(
        "AI-powered interior design pipeline.\n\n"
        "**Route groups:**\n"
        "- `/auth` — register, login, logout, current user\n"
        "- `/account` — saved layouts and generated render images per user\n"
        "- `/inventory` — browse and search the furniture/asset catalog\n"
        "- `/pipeline` — run the design pipeline and poll results\n"
        "- `/coordinates` — normalize frontend coordinates before pipeline input\n"
    ),
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def _startup_logging() -> None:
    # Ensure our internal logs (agents/RAG/search) show up when running via uvicorn.
    setup_logging()
    _initialize_runtime_schema()
    _log_runtime_profile()


def _initialize_runtime_schema() -> None:
    try:
        ensure_runtime_schema()
        loaded_count = ensure_demo_inventory_loaded()
        if loaded_count:
            logger.info("Loaded %d bundled demo inventory assets.", loaded_count)
    except Exception as exc:
        logger.exception("Runtime schema initialization skipped: %s", exc)


def _log_runtime_profile() -> None:
    gemini_config = root_config.services.gemini
    mistral_config = root_config.services.mistral
    ollama_config = root_config.services.ollama
    semantic_enabled = root_config.services.semantic_search.enabled
    logged_provider = False

    if mistral_config.enabled and mistral_config.models is not None:
        logger.info(
            "LLM provider: Mistral | base_url=%s | primary=%s | helper=%s | embedding=%s",
            mistral_config.base_url,
            mistral_config.models.primary.name,
            mistral_config.models.helper.name,
            mistral_config.models.embedding.name,
        )
        logged_provider = True
    if ollama_config.enabled and ollama_config.models is not None:
        logger.info(
            "LLM provider: Ollama | base_url=%s | primary=%s | helper=%s | embedding=%s",
            ollama_config.base_url,
            ollama_config.models.primary.name,
            ollama_config.models.helper.name,
            ollama_config.models.embedding.name,
        )
        logged_provider = True
    if gemini_config.enabled and gemini_config.models is not None:
        logger.info(
            "LLM provider: Gemini | base_url=%s | primary=%s | helper=%s | embedding=%s",
            gemini_config.base_url,
            gemini_config.models.primary.name,
            gemini_config.models.helper.name,
            gemini_config.models.embedding.name,
        )
        logged_provider = True
    if not logged_provider and OpenAIConfig.IS_OPENAI_AZURE:
        primary_model, helper_model, embedding_model = _openai_model_names(
            OpenAIConfig.MODELS
        )
        logger.info(
            "LLM provider: Azure OpenAI | primary=%s | helper=%s | embedding=%s",
            primary_model,
            helper_model,
            embedding_model,
        )
    elif not logged_provider:
        primary_model, helper_model, embedding_model = _openai_model_names(
            OpenAIConfig.MODELS
        )
        logger.info(
            "LLM provider: OpenAI-compatible API | primary=%s | helper=%s | embedding=%s",
            primary_model,
            helper_model,
            embedding_model,
        )

    if semantic_enabled:
        logger.info("Knowledge retrieval: semantic search enabled.")
    else:
        logger.info(
            "Knowledge retrieval: semantic search disabled; lexical fallback search enabled."
        )


def _openai_model_names(models: OpenAIModelGroupConfig) -> tuple[str, str, str]:
    resolved_models = models.resolved_models()
    return (
        resolved_models["primary"].name,
        resolved_models["helper"].name,
        resolved_models["embedding"].name,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(auth.router)
app.include_router(account.router)
app.include_router(inventory.router)
app.include_router(coordinates.router)

assets_gen_dir = Path(__file__).resolve().parents[1] / "assets_gen"
if assets_gen_dir.exists():
    app.mount(
        "/assets_gen",
        StaticFiles(directory=assets_gen_dir),
        name="assets_gen",
    )


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}
