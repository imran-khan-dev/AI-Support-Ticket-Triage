"""Knowledge base service layer.

Wraps Retrieval Stack components (DocumentIngestor, ChunkingService, BM25Searcher, DenseSearcher)
with strict tenant isolation. Each tenant gets its own in-memory BM25 and dense vector indexes.
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
from retrieval_stack.dense.searcher import DenseSearcher
from retrieval_stack.dense.index import VectorIndex
from retrieval_stack.dense.embeddings import SentenceTransformerEmbedding
from retrieval_stack.hybrid.searcher import HybridSearcher
from retrieval_stack.reranking.reranker import Reranker
from retrieval_stack.reranking.provider import SentenceTransformerCrossEncoder


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

    Each tenant has its own BM25 index and dense vector index built from their chunks.
    Indexes are rebuilt on demand when chunks change.
    """

    def __init__(
        self,
        db: Database,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        dense_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranking: bool = False,
    ):
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
        self._dense_model_name = dense_model_name
        self._reranker_model_name = reranker_model_name
        self._enable_reranking = enable_reranking
        self._dense_embedder: SentenceTransformerEmbedding | None = None
        self._reranker: Reranker | None = None
        self._indexes: dict[str, InvertedIndex] = {}
        self._searchers: dict[str, BM25Searcher] = {}
        self._dense_indexes: dict[str, VectorIndex] = {}
        self._dense_searchers: dict[str, DenseSearcher] = {}
        self._hybrid_searchers: dict[str, HybridSearcher] = {}

    def _get_dense_embedder(self) -> SentenceTransformerEmbedding:
        if self._dense_embedder is None:
            self._dense_embedder = SentenceTransformerEmbedding(self._dense_model_name)
        return self._dense_embedder

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            provider = SentenceTransformerCrossEncoder(self._reranker_model_name)
            self._reranker = Reranker(provider)
        return self._reranker

    def _get_index(self, tenant_id: str) -> InvertedIndex:
        if tenant_id in self._indexes:
            return self._indexes[tenant_id]

        index = InvertedIndex(self._tokenizer)
        chunks = get_all_chunks_for_tenant(self._db, tenant_id)
        for chunk in chunks:
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
        if tenant_id in self._searchers:
            return self._searchers[tenant_id]

        index = self._get_index(tenant_id)
        searcher = BM25Searcher(index, self._scorer, self._tokenizer)
        self._searchers[tenant_id] = searcher
        return searcher

    def _get_dense_index(self, tenant_id: str) -> VectorIndex:
        if tenant_id in self._dense_indexes:
            return self._dense_indexes[tenant_id]

        index = VectorIndex()
        chunks = get_all_chunks_for_tenant(self._db, tenant_id)
        if chunks:
            embedder = self._get_dense_embedder()
            embeddings = embedder.embed_batch([chunk.content for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings):
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
                index.add(rs_chunk, embedding)
        self._dense_indexes[tenant_id] = index
        return index

    def _get_dense_searcher(self, tenant_id: str) -> DenseSearcher:
        if tenant_id in self._dense_searchers:
            return self._dense_searchers[tenant_id]

        index = self._get_dense_index(tenant_id)
        embedder = self._get_dense_embedder()
        searcher = DenseSearcher(index, embedder)
        self._dense_searchers[tenant_id] = searcher
        return searcher

    def _get_hybrid_searcher(self, tenant_id: str) -> HybridSearcher:
        if tenant_id in self._hybrid_searchers:
            return self._hybrid_searchers[tenant_id]

        bm25_searcher = self._get_searcher(tenant_id)
        dense_searcher = self._get_dense_searcher(tenant_id)
        searcher = HybridSearcher(bm25_searcher, dense_searcher, rrf_k=60)
        self._hybrid_searchers[tenant_id] = searcher
        return searcher

    def invalidate_index(self, tenant_id: str) -> None:
        self._indexes.pop(tenant_id, None)
        self._searchers.pop(tenant_id, None)
        self._dense_indexes.pop(tenant_id, None)
        self._dense_searchers.pop(tenant_id, None)
        self._hybrid_searchers.pop(tenant_id, None)

    def _apply_reranking(
        self,
        tenant_id: str,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Apply cross-encoder reranking to search results."""
        if not results:
            return results

        # Convert our SearchResult objects to objects with chunk attribute for the Reranker
        class RerankCandidate:
            def __init__(self, search_result: SearchResult):
                self.chunk_id = search_result.chunk.id
                self.chunk = RSChunk(
                    id=search_result.chunk.id,
                    document_id=search_result.chunk.document_id,
                    index=search_result.chunk.chunk_index,
                    content=search_result.chunk.content,
                    token_count=search_result.chunk.token_count,
                    metadata=search_result.chunk.metadata,
                    start_char=search_result.chunk.start_char,
                    end_char=search_result.chunk.end_char,
                )
                self.score = search_result.score

        candidates = [RerankCandidate(r) for r in results]

        reranker = self._get_reranker()
        reranked_results = reranker.rerank(query, candidates, top_k=top_k)

        # Convert back to SearchResult with reranked scores
        final_results = []
        for rr in reranked_results:
            chunks = get_all_chunks_for_tenant(self._db, tenant_id)
            chunk_map = {c.id: c for c in chunks}
            if rr.chunk_id in chunk_map:
                final_results.append(SearchResult(chunk=chunk_map[rr.chunk_id], score=rr.relevance_score))

        return final_results

    def _ingest_bytes(self, file_content: bytes, filename: str, content_type: str) -> str:
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
        checksum = hashlib.sha256(file_content).hexdigest()
        byte_size = len(file_content)

        doc = create_document(
            self._db,
            tenant_id=tenant_id,
            filename=filename,
            content_type=content_type,
            byte_size=byte_size,
            checksum=checksum,
        )

        update_document_status(self._db, tenant_id, doc.id, "processing")

        try:
            text = self._ingest_bytes(file_content, filename, content_type)
            chunks = self._chunker.chunk_text(doc.id, text)

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

            stored_chunks = create_chunks(
                self._db,
                tenant_id=tenant_id,
                document_id=doc.id,
                chunks=chunk_dicts,
            )

            updated_doc = update_document_status(self._db, tenant_id, doc.id, "completed")
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
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the tenant's knowledge base using BM25 retrieval."""
        searcher = self._get_searcher(tenant_id)
        results = searcher.search(query, top_k=top_k)

        search_results = []
        for result in results:
            chunks = get_all_chunks_for_tenant(self._db, tenant_id)
            chunk_map = {c.id: c for c in chunks}
            if result.chunk_id in chunk_map:
                search_results.append(SearchResult(chunk=chunk_map[result.chunk_id], score=result.score))

        if rerank and self._enable_reranking:
            search_results = self._apply_reranking(tenant_id, query, search_results, top_k)

        return search_results

    def search_dense(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the tenant's knowledge base using dense vector retrieval."""
        searcher = self._get_dense_searcher(tenant_id)
        results = searcher.search(query, top_k=top_k)

        search_results = []
        for result in results:
            chunks = get_all_chunks_for_tenant(self._db, tenant_id)
            chunk_map = {c.id: c for c in chunks}
            if result.chunk_id in chunk_map:
                search_results.append(SearchResult(chunk=chunk_map[result.chunk_id], score=result.score))

        if rerank and self._enable_reranking:
            search_results = self._apply_reranking(tenant_id, query, search_results, top_k)

        return search_results

    def search_hybrid(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the tenant's knowledge base using hybrid (BM25 + dense) retrieval with RRF."""
        searcher = self._get_hybrid_searcher(tenant_id)
        results = searcher.search(query, top_k=top_k)

        search_results = []
        for result in results:
            chunks = get_all_chunks_for_tenant(self._db, tenant_id)
            chunk_map = {c.id: c for c in chunks}
            if result.chunk_id in chunk_map:
                search_results.append(SearchResult(chunk=chunk_map[result.chunk_id], score=result.rrf_score))

        if rerank and self._enable_reranking:
            search_results = self._apply_reranking(tenant_id, query, search_results, top_k)

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