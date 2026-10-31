"""Database provider factory connecting Settings to SQLiteRepository or SupabaseRepository."""

from app.core.config import Settings
from app.repositories.base import BaseRepository
from app.repositories.sqlite import SQLiteRepository
from app.repositories.supabase import SupabaseRepository


def create_database_repository(settings: Settings) -> BaseRepository:
    """Create the configured database repository instance."""
    if settings.database_provider == "supabase":
        return SupabaseRepository(
            supabase_url=settings.supabase_url,
            service_role_key=settings.supabase_service_role_key,
            vector_dimensions=settings.supabase_vector_dimensions,
        )
    return SQLiteRepository(db_path=settings.sqlite_db_path)
