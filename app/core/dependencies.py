"""Single composition root and dependency container for Orbit AI.

Initializes Settings, Providers, Services, RateLimiter, and SessionManager in one
authoritative place without duplicating provider initialization across files.
"""

from typing import Any, Dict, Optional
from app.core.config import Settings, load_settings
from app.core.logging import configure_logging
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.core.security import AdminSessionManager
from app.providers.database import create_database_repository
from app.providers.embeddings import (
    BaseEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.providers.llm import (
    BaseLlmProvider,
    LocalLlmProvider,
    OpenAICompatibleLlmProvider,
)
from app.providers.notifications import NotificationProvider
from app.providers.storage import (
    BaseStorageProvider,
    LocalStorageProvider,
    SupabaseStorageProvider,
)
from app.providers.tts import TtsProvider
from app.providers.vector import (
    BaseVectorProvider,
    LocalVectorProvider,
    QdrantVectorProvider,
    SupabaseVectorProvider,
)
from app.repositories.base import BaseRepository
from app.services.advanced_rag import AdvancedRagService
from app.services.analytics import AnalyticsService
from app.services.chat import ChatService
from app.services.clients import ClientService
from app.services.conversations import ConversationService
from app.services.grounding import GroundingService
from app.services.knowledge import KnowledgeService
from app.services.leads import LeadService
from app.services.notifications import NotificationService
from app.services.parser import DocumentParserService
from app.services.prompts import PromptService
from app.utils.pii import mask_secret_preview


class ServiceContainer:
    """Centralized composition root holding all initialized providers and services."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or load_settings()
        self.logger = configure_logging(
            level="DEBUG" if self.settings.orbit_rag_diagnostics else "INFO",
            extra_secrets=self.settings.known_secrets(),
        )
        self.rate_limiter = SlidingWindowRateLimiter(window_seconds=60)
        self.session_manager = AdminSessionManager(secret_key=self.settings.admin_api_key)

        # 1. Database Repository
        self.repository: BaseRepository = create_database_repository(self.settings)

        # 2. Storage Provider
        if self.settings.storage_provider == "supabase":
            self.storage_provider: BaseStorageProvider = SupabaseStorageProvider(
                supabase_url=self.settings.supabase_url,
                service_role_key=self.settings.supabase_service_role_key,
                bucket=self.settings.supabase_storage_bucket,
            )
        else:
            self.storage_provider = LocalStorageProvider(
                base_dir=self.settings.local_storage_dir,
            )

        # 3. Embedding Provider
        if self.settings.embedding_provider == "openai-compatible":
            self.embedding_provider: BaseEmbeddingProvider = OpenAICompatibleEmbeddingProvider(
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
                model=self.settings.embedding_model,
                dimensions=self.settings.embedding_dimensions,
            )
        else:
            self.embedding_provider = LocalEmbeddingProvider(
                dimensions=self.settings.embedding_dimensions,
            )

        # 4. Vector Provider
        if self.settings.vector_provider == "supabase":
            self.vector_provider: BaseVectorProvider = SupabaseVectorProvider(
                supabase_url=self.settings.supabase_url,
                service_role_key=self.settings.supabase_service_role_key,
                dimensions=self.settings.supabase_vector_dimensions,
            )
        elif self.settings.vector_provider == "qdrant":
            self.vector_provider = QdrantVectorProvider(
                qdrant_url=self.settings.qdrant_url,
                collection_name=self.settings.qdrant_collection,
                api_key=self.settings.qdrant_api_key,
                dimensions=self.settings.supabase_vector_dimensions,
            )
        else:
            self.vector_provider = LocalVectorProvider(
                repository=self.repository,
                dimensions=self.settings.supabase_vector_dimensions,
            )

        # 5. LLM Provider
        if self.settings.llm_provider_type == "openai-compatible":
            self.llm_provider: BaseLlmProvider = OpenAICompatibleLlmProvider(
                provider_name=self.settings.llm_provider_name,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                timeout_ms=self.settings.llm_timeout_ms,
                max_retries=self.settings.llm_max_retries,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                chat_path=self.settings.llm_chat_path,
                health_path=self.settings.llm_health_path,
                auth_header_name=self.settings.llm_auth_header_name,
                auth_scheme=self.settings.llm_auth_scheme,
                extra_headers=self.settings.llm_headers,
                query_params=self.settings.llm_query_params,
            )
        else:
            self.llm_provider = LocalLlmProvider(model_name=self.settings.llm_model)

        # 6. Notification Provider
        self.notification_provider = NotificationProvider(
            repository=self.repository,
            provider_type=self.settings.notification_provider,
            smtp_host=self.settings.smtp_host,
            smtp_port=self.settings.smtp_port,
            smtp_user=self.settings.smtp_user,
            smtp_pass=self.settings.smtp_pass,
            smtp_from=self.settings.smtp_from,
        )

        # 7. Optional TTS Provider (server-only)
        self.tts_provider = TtsProvider(self.settings)

        # 8. Domain Services
        self.parser_service = DocumentParserService(
            max_upload_size_mb=self.settings.max_upload_size_mb
        )
        self._bind_services()
        self._initialized = False
        self.supabase_migration_pending = False
        self.demo_public_keys: Dict[str, str] = {}

    def _bind_services(self) -> None:
        self.notification_provider.repository = self.repository
        self.client_service = ClientService(
            repository=self.repository,
            storage=self.storage_provider,
        )
        self.knowledge_service = KnowledgeService(
            repository=self.repository,
            storage_provider=self.storage_provider,
            embedding_provider=self.embedding_provider,
            parser=self.parser_service,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.rag_service = AdvancedRagService(
            repository=self.repository,
            embedding_provider=self.embedding_provider,
            vector_provider=self.vector_provider,
            llm_provider=self.llm_provider,
            diagnostics_enabled=self.settings.orbit_rag_diagnostics,
        )
        self.grounding_service = GroundingService()
        self.conversation_service = ConversationService(
            repository=self.repository,
            llm_provider=self.llm_provider,
            history_limit=self.settings.conversation_history_limit,
        )
        self.chat_service = ChatService(
            repository=self.repository,
            conversation_service=self.conversation_service,
            rag_service=self.rag_service,
            grounding_service=self.grounding_service,
            llm_provider=self.llm_provider,
            max_message_length=self.settings.max_message_length,
        )
        self.lead_service = LeadService(
            repository=self.repository,
            notification_provider=self.notification_provider,
        )
        self.notification_service = NotificationService(
            repository=self.repository,
            provider=self.notification_provider,
        )
        self.prompt_service = PromptService(
            repository=self.repository,
            rag_service=self.rag_service,
            grounding_service=self.grounding_service,
            llm_provider=self.llm_provider,
        )
        self.analytics_service = AnalyticsService(repository=self.repository)

    async def initialize(self, seed_demo_data: bool = False) -> None:
        """Initialize persistence layer and optionally seed realistic local demo tenants."""
        if self._initialized:
            return
        await self.repository.initialize()

        # If Supabase is configured but 0001_initial.sql has not yet been applied in the Supabase SQL Editor,
        # keep the platform operational via local SQLite + LocalVectorProvider until migration is applied.
        if self.settings.database_provider == "supabase" and getattr(self.repository, "_http_handler", None) is None:
            db_check = await self.repository.check_health()
            if not db_check.get("ok"):
                from app.repositories.sqlite import SQLiteRepository

                self.logger.warning(
                    "Supabase schema not yet migrated (0001_initial.sql pending); activating local SQLite fallback."
                )
                self.supabase_migration_pending = True
                self.repository = SQLiteRepository(db_path=self.settings.sqlite_db_path)
                await self.repository.initialize()
                if self.settings.vector_provider == "supabase":
                    self.vector_provider = LocalVectorProvider(
                        repository=self.repository,
                        dimensions=self.settings.supabase_vector_dimensions,
                    )
                self._bind_services()

        self._initialized = True
        if seed_demo_data or self.supabase_migration_pending:
            existing = await self.repository.list_clients()
            slugs = {t.get("slug") for t in existing}
            if not existing or "apex-fitness" not in slugs:
                try:
                    await self._seed_initial_tenants()
                    existing = await self.repository.list_clients()
                except Exception:
                    pass
            for t in existing:
                if t["id"] not in self.demo_public_keys:
                    try:
                        raw_k, _ = await self.client_service.generate_api_key(
                            t["id"], name="Sandbox Preview Key"
                        )
                        self.demo_public_keys[t["id"]] = raw_k
                    except Exception:
                        pass

    async def _seed_initial_tenants(self) -> None:
        """Seed two realistic multi-tenant examples for immediate local testing."""
        # Tenant 1: Apex Fitness & Performance Club
        gym_tenant, gym_key = await self.client_service.create_tenant(
            {
                "business_name": "Apex Performance Club",
                "slug": "apex-fitness",
                "company_name": "Apex Performance & Wellness Ltd.",
                "assistant_name": "Apex Concierge",
                "welcome_message": "Welcome to Apex Performance Club! Ask me about memberships, class schedules, or personal training.",
                "system_prompt": (
                    "You are the Apex Concierge for Apex Performance Club. "
                    "Answer member questions accurately using only verified club documents."
                ),
                "fallback_answer": (
                    "I don't have verified details on that in our club handbook. "
                    "Please leave your contact details below and a Membership Advisor will reach out within 2 hours."
                ),
                "domains": ["localhost", "127.0.0.1", "apexfitness.example.com"],
                "widget_config": {
                    "primary_color": "#2563eb",
                    "theme_mode": "light",
                    "position": "bottom-right",
                    "width": 380,
                    "height": 560,
                    "border_radius": 14,
                    "input_placeholder": "Ask about memberships, hours, or training...",
                    "show_branding": True,
                    "suggested_questions": [
                        "What are your membership plans and pricing?",
                        "What are the club opening hours?",
                        "How does the cancellation policy work?",
                    ],
                },
            }
        )
        if gym_key:
            self.demo_public_keys[gym_tenant["id"]] = gym_key

        gym_handbook = (
            "Apex Performance Club — Official Member Handbook (2026)\n\n"
            "Membership Plans and Pricing:\n"
            "Our Core Membership is $59 per month and includes unlimited access to the strength floor, cardio zone, and locker rooms. "
            "Our Peak All-Access Membership is $99 per month and includes unlimited group HIIT classes, Olympic lifting platforms, sauna & cold plunge recovery suite, and 2 guest passes per month.\n\n"
            "Opening Hours and Schedule:\n"
            "Apex Performance Club is open Monday through Friday from 5:00am to 11:00pm, and Saturday & Sunday from 6:00am to 9:00pm.\n\n"
            "Cancellation and Freeze Policy:\n"
            "Members may freeze their membership for up to 3 months per year at no fee. "
            "Cancellations require 14 days notice prior to the monthly billing date via support@apexperformance.example.com or by calling +1-555-234-5678."
        ).encode("utf-8")
        await self.knowledge_service.ingest_document(
            tenant_id=gym_tenant["id"],
            filename="apex_member_handbook_2026.txt",
            content=gym_handbook,
            content_type="text/plain",
        )

        # Create a sample conversation and lead for Tenant 1
        chat_res = await self.chat_service.process_chat(
            tenant=gym_tenant,
            message="What are your membership plans and pricing?",
            session_id="demo-session-apex-01",
        )
        await self.lead_service.submit_lead(
            tenant=gym_tenant,
            payload={
                "name": "Elena Rostova",
                "email": "elena.rostova@example.com",
                "phone": "+1-555-876-5432",
                "requirement": "Peak All-Access Membership + Personal Training Assessment",
                "message": "Interested in joining next Monday and touring the recovery suite.",
                "conversationId": chat_res["conversationId"],
            },
        )

        # Tenant 2: CloudScale B2B SaaS Support
        b2b_tenant, b2b_key = await self.client_service.create_tenant(
            {
                "business_name": "CloudScale Enterprise",
                "slug": "cloudscale-b2b",
                "company_name": "CloudScale Infrastructure Inc.",
                "assistant_name": "CloudScale Support AI",
                "welcome_message": "Hello! Ask me about CloudScale SLA guarantees, SOC2 compliance, or enterprise API limits.",
                "domains": ["localhost", "127.0.0.1", "cloudscale.example.com"],
                "widget_config": {
                    "primary_color": "#0f766e",
                    "theme_mode": "dark",
                    "position": "bottom-right",
                    "width": 390,
                    "height": 580,
                    "border_radius": 12,
                    "suggested_questions": [
                        "What is the enterprise uptime SLA?",
                        "What security certifications do you hold?",
                    ],
                },
            }
        )
        if b2b_key:
            self.demo_public_keys[b2b_tenant["id"]] = b2b_key

        b2b_doc = (
            "CloudScale Enterprise SLA & Security Specification\n\n"
            "Enterprise Uptime SLA:\n"
            "CloudScale guarantees 99.99% monthly uptime for Enterprise tier deployments with automated multi-region failover in under 15 seconds.\n\n"
            "Security & Compliance:\n"
            "CloudScale holds SOC 2 Type II, ISO 27001, and GDPR compliance certifications. All data at rest is encrypted with AES-256."
        ).encode("utf-8")
        await self.knowledge_service.ingest_document(
            tenant_id=b2b_tenant["id"],
            filename="cloudscale_sla_security.md",
            content=b2b_doc,
            content_type="text/markdown",
        )

    async def get_provider_diagnostics(self) -> Dict[str, Any]:
        """Collect safe diagnostics across all providers (never exposes full secrets)."""
        if self.supabase_migration_pending and self.settings.supabase_url and self.settings.supabase_service_role_key:
            from app.repositories.supabase import SupabaseRepository

            candidate_repo = SupabaseRepository(
                supabase_url=self.settings.supabase_url,
                service_role_key=self.settings.supabase_service_role_key,
                vector_dimensions=self.settings.supabase_vector_dimensions,
            )
            probe = await candidate_repo.check_health()
            if probe.get("ok"):
                self.supabase_migration_pending = False
                self.repository = candidate_repo
                if self.settings.vector_provider == "supabase":
                    self.vector_provider = SupabaseVectorProvider(
                        supabase_url=self.settings.supabase_url,
                        service_role_key=self.settings.supabase_service_role_key,
                        dimensions=self.settings.supabase_vector_dimensions,
                    )
                self._bind_services()

        db_health = await self.repository.check_health()
        storage_health = await self.storage_provider.check_health()
        vector_health = await self.vector_provider.check_health()
        emb_health = await self.embedding_provider.check_health()
        llm_health = await self.llm_provider.check_health()
        notif_health = await self.notification_provider.check_health()
        tts_health = await self.tts_provider.check_health()

        return {
            "environment": self.settings.app_env,
            "database": {
                **db_health,
                "supabase_url": self.settings.supabase_url or "local-sqlite",
                "service_role_key_preview": mask_secret_preview(self.settings.supabase_service_role_key),
                "supabase_migration_pending": self.supabase_migration_pending,
            },
            "storage": {
                **storage_health,
                "bucket": self.settings.supabase_storage_bucket,
            },
            "vector": {
                **vector_health,
                "configured_dimensions": self.settings.supabase_vector_dimensions,
            },
            "embeddings": {
                **emb_health,
                "api_key_preview": mask_secret_preview(self.settings.embedding_api_key),
            },
            "llm": {
                **llm_health,
                "base_url": self.settings.llm_base_url or "local-in-process",
                "api_key_preview": mask_secret_preview(self.settings.llm_api_key),
                "timeout_ms": self.settings.llm_timeout_ms,
                "max_retries": self.settings.llm_max_retries,
            },
            "notifications": notif_health,
            "tts": tts_health,
            "rag_diagnostics_enabled": self.settings.orbit_rag_diagnostics,
        }
