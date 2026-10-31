"""Application lifespan management for FastAPI startup and shutdown."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from app.core.dependencies import ServiceContainer


@asynccontextmanager
async def app_lifespan(app: Any) -> AsyncGenerator[None, None]:
    """Initialize the ServiceContainer on startup and attach to app.state."""
    container: ServiceContainer = getattr(app.state, "container", None) or ServiceContainer()
    seed_demo = container.settings.app_env == "development"
    await container.initialize(seed_demo_data=seed_demo)
    app.state.container = container
    yield
