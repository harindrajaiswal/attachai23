"""Safe versioned knowledge ingestion and reindexing service.

Implements Section 10 & Fixes Defect #10:
UPLOAD NEW FILE -> VALIDATE FILE -> STORE PRIVATE OBJECT -> PARSE DOCUMENT
-> CREATE CHUNKS -> GENERATE EMBEDDINGS -> WRITE STAGING VECTOR VERSION
-> VERIFY INDEX -> ACTIVATE NEW VERSION -> CLEAN OLD VERSION.

If re-ingestion of an existing source fails at any stage, the previous active
index version remains intact and ready for customer queries.
"""

import uuid
from typing import Any, Dict, List, Optional
from app.core.exceptions import NotFoundError, OrbitError, ValidationError
from app.domain.enums import KnowledgeStatus
from app.providers.embeddings import BaseEmbeddingProvider
from app.providers.storage import BaseStorageProvider
from app.repositories.base import BaseRepository
from app.services.parser import DocumentParserService
from app.utils.chunking import chunk_text
from app.utils.pii import redact_text
from app.utils.validators import build_storage_object_path, sanitize_filename


class KnowledgeService:
    """Manages tenant-isolated document upload, versioned indexing, and safe rebuilds."""

    def __init__(
        self,
        repository: BaseRepository,
        storage_provider: BaseStorageProvider,
        embedding_provider: BaseEmbeddingProvider,
        parser: DocumentParserService,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ) -> None:
        self.repository = repository
        self.storage = storage_provider
        self.embeddings = embedding_provider
        self.parser = parser
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest_document(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """Ingest a new document or safely replace an existing document with the same name."""
        client = await self.repository.get_client(tenant_id)
        if not client:
            raise NotFoundError(f"Tenant '{tenant_id}' not found.")

        safe_name = sanitize_filename(filename)
        existing_file = await self.repository.find_knowledge_file_by_name(tenant_id, filename)

        if existing_file and existing_file.get("status") == KnowledgeStatus.READY.value:
            # Safe versioned replacement: preserve active_version if the new ingestion fails
            source_id = existing_file["id"]
            previous_active_version = int(existing_file.get("active_version", 1))
            staging_version = previous_active_version + 1
            storage_path = build_storage_object_path(
                tenant_id, source_id, f"v{staging_version}_{safe_name}"
            )
            try:
                parsed_text = self.parser.parse_document(filename, content, content_type)
                await self.storage.upload(storage_path, content, content_type=content_type)
                raw_chunks = chunk_text(parsed_text, self.chunk_size, self.chunk_overlap)
                if not raw_chunks:
                    raise ValidationError("Document produced zero chunks.")
                vectors = await self.embeddings.embed_texts(raw_chunks)

                chunk_records = [
                    {
                        "id": str(uuid.uuid4()),
                        "source_filename": filename,
                        "chunk_index": idx,
                        "content": chunk_str,
                        "embedding": vec,
                    }
                    for idx, (chunk_str, vec) in enumerate(zip(raw_chunks, vectors))
                ]
                await self.repository.insert_staging_vectors(
                    tenant_id=tenant_id,
                    source_id=source_id,
                    version=staging_version,
                    chunks=chunk_records,
                )
                verified_count = await self.repository.count_vectors_for_version(
                    tenant_id, source_id, staging_version
                )
                if verified_count != len(chunk_records) or verified_count == 0:
                    await self.repository.delete_vectors_for_version(tenant_id, source_id, staging_version)
                    raise OrbitError("Staging vector index verification failed.", status_code=500)

                # Activate new version and clean up old version
                await self.repository.activate_vector_version(tenant_id, source_id, staging_version)
                old_storage_path = existing_file.get("storage_path")
                updated = await self.repository.update_knowledge_file(
                    tenant_id,
                    source_id,
                    {
                        "safe_filename": safe_name,
                        "storage_path": storage_path,
                        "mime_type": content_type,
                        "file_size": len(content),
                        "status": KnowledgeStatus.READY.value,
                        "active_version": staging_version,
                        "chunk_count": verified_count,
                        "error_message": None,
                    },
                )
                if old_storage_path and old_storage_path != storage_path:
                    await self.storage.delete(old_storage_path)
                assert updated is not None
                return updated

            except Exception as exc:
                # Clean any partial staging artifacts; keep previous_active_version active!
                await self.repository.delete_vectors_for_version(tenant_id, source_id, staging_version)
                safe_err = redact_text(str(exc))
                await self.repository.update_knowledge_file(
                    tenant_id,
                    source_id,
                    {
                        "status": KnowledgeStatus.READY.value,
                        "active_version": previous_active_version,
                        "error_message": f"Re-upload failed (preserved v{previous_active_version}): {safe_err}",
                    },
                )
                raise

        # Fresh document upload
        source_id = str(uuid.uuid4())
        storage_path = build_storage_object_path(tenant_id, source_id, safe_name)

        # Validate file metadata & content BEFORE storing or marking ready
        parsed_text = self.parser.parse_document(filename, content, content_type)
        raw_chunks = chunk_text(parsed_text, self.chunk_size, self.chunk_overlap)
        if not raw_chunks:
            raise ValidationError("Document contains no indexable text chunks.")

        file_record = await self.repository.create_knowledge_file(
            {
                "id": source_id,
                "tenant_id": tenant_id,
                "original_filename": filename,
                "safe_filename": safe_name,
                "storage_path": storage_path,
                "mime_type": content_type,
                "file_size": len(content),
                "status": KnowledgeStatus.PROCESSING.value,
                "active_version": 1,
                "chunk_count": 0,
                "error_message": None,
            }
        )

        try:
            await self.storage.upload(storage_path, content, content_type=content_type)
            vectors = await self.embeddings.embed_texts(raw_chunks)
            chunk_records = [
                {
                    "id": str(uuid.uuid4()),
                    "source_filename": filename,
                    "chunk_index": idx,
                    "content": chunk_str,
                    "embedding": vec,
                }
                for idx, (chunk_str, vec) in enumerate(zip(raw_chunks, vectors))
            ]
            await self.repository.insert_staging_vectors(
                tenant_id=tenant_id,
                source_id=source_id,
                version=1,
                chunks=chunk_records,
            )
            verified_count = await self.repository.count_vectors_for_version(tenant_id, source_id, 1)
            if verified_count != len(chunk_records) or verified_count == 0:
                raise OrbitError("Vector verification failed during initial ingestion.", status_code=500)

            await self.repository.activate_vector_version(tenant_id, source_id, 1)
            updated = await self.repository.update_knowledge_file(
                tenant_id,
                source_id,
                {
                    "status": KnowledgeStatus.READY.value,
                    "active_version": 1,
                    "chunk_count": verified_count,
                    "error_message": None,
                },
            )
            assert updated is not None
            return updated

        except Exception as exc:
            await self.repository.delete_vectors_for_version(tenant_id, source_id, 1)
            safe_err = redact_text(str(exc))
            await self.repository.update_knowledge_file(
                tenant_id,
                source_id,
                {
                    "status": KnowledgeStatus.FAILED.value,
                    "error_message": safe_err,
                },
            )
            raise

    async def rebuild_document(self, tenant_id: str, file_id: str) -> Dict[str, Any]:
        """Safely rebuild a single knowledge document's chunks and embeddings."""
        file_rec = await self.repository.get_knowledge_file(tenant_id, file_id)
        if not file_rec:
            raise NotFoundError("Knowledge file not found for this tenant.")

        prev_version = int(file_rec.get("active_version", 1))
        next_version = prev_version + 1

        try:
            raw_bytes = await self.storage.download(file_rec["storage_path"])
            parsed_text = self.parser.parse_document(
                file_rec["original_filename"],
                raw_bytes,
                file_rec.get("mime_type", "text/plain"),
            )
            raw_chunks = chunk_text(parsed_text, self.chunk_size, self.chunk_overlap)
            vectors = await self.embeddings.embed_texts(raw_chunks)
            chunk_records = [
                {
                    "id": str(uuid.uuid4()),
                    "source_filename": file_rec["original_filename"],
                    "chunk_index": idx,
                    "content": chunk_str,
                    "embedding": vec,
                }
                for idx, (chunk_str, vec) in enumerate(zip(raw_chunks, vectors))
            ]
            await self.repository.insert_staging_vectors(
                tenant_id=tenant_id,
                source_id=file_id,
                version=next_version,
                chunks=chunk_records,
            )
            verified_count = await self.repository.count_vectors_for_version(
                tenant_id, file_id, next_version
            )
            if verified_count != len(chunk_records) or verified_count == 0:
                await self.repository.delete_vectors_for_version(tenant_id, file_id, next_version)
                raise OrbitError("Staging index verification failed during rebuild.", status_code=500)

            await self.repository.activate_vector_version(tenant_id, file_id, next_version)
            updated = await self.repository.update_knowledge_file(
                tenant_id,
                file_id,
                {
                    "status": KnowledgeStatus.READY.value,
                    "active_version": next_version,
                    "chunk_count": verified_count,
                    "error_message": None,
                },
            )
            assert updated is not None
            return updated
        except Exception as exc:
            await self.repository.delete_vectors_for_version(tenant_id, file_id, next_version)
            safe_err = redact_text(str(exc))
            await self.repository.update_knowledge_file(
                tenant_id,
                file_id,
                {
                    "error_message": f"Rebuild failed (active v{prev_version} preserved): {safe_err}",
                },
            )
            raise

    async def list_files(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.repository.list_knowledge_files(tenant_id=tenant_id)

    async def delete_file(self, tenant_id: str, file_id: str) -> bool:
        file_rec = await self.repository.get_knowledge_file(tenant_id, file_id)
        if not file_rec:
            raise NotFoundError("Knowledge file not found.")
        if file_rec.get("storage_path"):
            await self.storage.delete(file_rec["storage_path"])
        return await self.repository.delete_knowledge_file(tenant_id, file_id)
