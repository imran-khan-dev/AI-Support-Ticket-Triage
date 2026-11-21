"""Knowledge base service layer.

Wraps Retrieval Stack components (DocumentIngestor, ChunkingService, BM25Searcher)
with strict tenant isolation. Each tenant gets its own in-memory BM25 index.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Literal

from app.database import Database
from app.knowledge.repository import (
    KnowledgeChunk,
    KnowledgeDocument,
    create_chunks,
    create_document,
    delete_document,
    get_all_chunks_for_tenant,
    get_chunks_for_document,
    get_document,
    list_documents,
    update_document_status,
)

# Retrieval Stack imports
from retrieval_stack.ingestion.service import DocumentIngestor
from retrieval_stack.ingestion.errors import IngestionError
from retrieval_stack.chunking.service import ChunkingService, ChunkConfig
from retrieval_stack.chunking.models import Chunk as RSChunk
from retrieval_stack.bm25.searcher import BM25Searcher
from retrieval_stack.bm25.index import InvertedIndex
from retrieval_stack.bm25.tokenizer import Tokenizer
from retrieval_stack.bm25.scorer import BM25Scorer


@dataclass(frozen=True)
class IngestionResult:
    document: KnowledgeDocument
    chunks_created: int


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float


class KnowledgeService:
    """Tenant-isolated knowledge base service.

    Each tenant has its own BM25 index built from their chunks.
    Indexes are rebuilt on demand when chunks change.
    """

    def __init__(self, db: Database, chunk_size: int = 512, chunk_overlap: int = 64):
        self._db = db
        self._chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding="cl100k_base",
        )
        self._ingestor = DocumentIngestor()
        self._chunker = ChunkingService(self._chunk_config)
        self._tokenizer = Tokenizer()
        self._scorer = BM25Scorer()
        self._indexes: dict[str, InvertedIndex] = {}
        self._searchers: dict[str, BM25Searcher] = {}

    def _get_index(self, tenant_id: str) -> InvertedIndex:
        """Get or build the BM25 index for a tenant."""
        if tenant_id in self._indexes:
            return self._indexes[tenant_id]

        index = InvertedIndex(self._tokenizer)
        chunks = get_all_chunks_for_tenant(self._db, tenant_id)
        for chunk in chunks:
            # Convert to Retrieval Stack Chunk model
            rs_chunk = RSChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            index.add_chunk(rs_chunk)
        self._indexes[tenant_id] = index
        return index

    def _get_searcher(self, tenant_id: str) -> BM25Searcher:
        """Get or create the BM25 searcher for a tenant."""
        if tenant_id in self._searchers:
            return self._searchers[tenant_id]

        index = self._get_index(tenant_id)
        searcher = BM25Searcher(index, self._scorer, self._tokenizer)
        self._searchers[tenant_id] = searcher
        return searcher

    def invalidate_index(self, tenant_id: str) -> None:
        """Invalidate the cached index for a tenant (call after ingestion)."""
        self._indexes.pop(tenant_id, None)
        self._searchers.pop(tenant_id, None)

    def _ingest_bytes(self, file_content: bytes, filename: str, content_type: str) -> str:
        """Ingest file content from bytes using Retrieval Stack's DocumentIngestor.

        The ingestor expects a file path, so we write to a temporary file.
        """
        # Determine file extension from content_type or filename
        suffix = ".txt"
        if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
            suffix = ".pdf"
        elif content_type == "text/plain" or filename.lower().endswith(".txt"):
            suffix = ".txt"
        elif content_type == "text/markdown" or filename.lower().endswith(".md"):
            suffix = ".md"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            doc = self._ingestor.ingest(tmp_path)
            return doc.content
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def ingest_document(
        self,
        tenant_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> IngestionResult:
        """Ingest a document: extract text, chunk, store, and index."""
        # Compute checksum
        checksum = hashlib.sha256(file_content).hexdigest()
        byte_size = len(file_content)

        # Create document record
        doc = create_document(
            self._db,
            tenant_id=tenant_id,
            filename=filename,
            content_type=content_type,
            byte_size=byte_size,
            checksum=checksum,
        )

        # Update status to processing
        update_document_status(self._db, tenant_id, doc.id, "processing")

        try:
            # Extract text using DocumentIngestor
            text = self._ingest_bytes(file_content, filename, content_type)

            # Chunk the text
            chunks = self._chunker.chunk_text(doc.id, text)

            # Convert to dict format for bulk insert
            chunk_dicts = [
                {
                    "chunk_index": c.index,
                    "content": c.content,
                    "token_count": c.token_count,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]

            # Store chunks
            stored_chunks = create_chunks(
                self._db,
                tenant_id=tenant_id,
                document_id=doc.id,
                chunks=chunk_dicts,
            )

            # Update document status to completed
            updated_doc = update_document_status(self._db, tenant_id, doc.id, "completed")

            # Invalidate index so it rebuilds with new chunks
            self.invalidate_index(tenant_id)

            return IngestionResult(document=updated_doc, chunks_created=len(stored_chunks))

        except IngestionError as e:
            update_document_status(
                self._db, tenant_id, doc.id, "failed", error_message=str(e)
            )
            raise
        except Exception as e:
            update_document_status(
                self._db, tenant_id, doc.id, "failed", error_message=str(e)
            )
            raise

    def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search chunks for a tenant using BM25."""
        searcher = self._get_searcher(tenant_id)
        results = searcher.search(query, top_k=top_k)

        # Fetch full chunk data from database
        search_results = []
        for result in results:
            # We need to fetch the chunk from DB
            chunks = get_all_chunks_for_tenant(self._db, tenant_id)
            chunk_map = {c.id: c for c in chunks}
            if result.chunk_id in chunk_map:
                search_results.append(SearchResult(chunk=chunk_map[result.chunk_id], score=result.score))

        return search_results

    def get_document(self, tenant_id: str, document_id: str) -> KnowledgeDocument | None:
        return get_document(self._db, tenant_id, document_id)

    def list_documents(
        self,
        tenant_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeDocument]:
        return list_documents(self._db, tenant_id, status=status, limit=limit, offset=offset)

    def get_document_chunks(self, tenant_id: str, document_id: str) -> list[KnowledgeChunk]:
        return get_chunks_for_document(self._db, tenant_id, document_id)

    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        result = delete_document(self._db, tenant_id, document_id)
        if result:
            self.invalidate_index(tenant_id)
        return result