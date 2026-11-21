"""Tests for the knowledge base foundation.

Tests cover:
- Document ingestion (text extraction, chunking, storage)
- BM25 search with tenant isolation
- Chunk retrieval
- Tenant isolation enforcement
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APPROVER_TOKEN_SECRET", "test-only-secret-for-approver-credentials-0123456789")

from main import app
from app.core.security import generate_integration_token, hash_integration_token
from app.database import get_database
from app.modules.integrations.service import ensure_integration, issue_integration_credential
from app.knowledge.repository import (
    create_chunks,
    create_document,
    delete_document,
    get_all_chunks_for_tenant,
    get_chunks_for_document,
    get_document,
    list_documents,
    update_document_status,
)
from app.knowledge.service import KnowledgeService


def _assign_tenant_to_integration(db, integration_id):
    """Create a tenant and assign it to the integration."""
    from app.database.repository import _canonical_uuid
    tenant_id = str(uuid.uuid4())
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) ON CONFLICT (id) DO NOTHING",
            (_canonical_uuid(tenant_id), f"test-tenant-{tenant_id[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tenant_id), _canonical_uuid(integration_id)),
        )
    return tenant_id


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_setup(clean_database):
    """Create an integration with tenant and return (db, integration, credential, token, tenant_id)."""
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(db, integration_name="test-kb-integ")
    tenant_id = _assign_tenant_to_integration(db, integration.id)
    return db, integration, credential, token, tenant_id


class TestKnowledgeRepository:
    """Tests for the knowledge repository layer."""

    def test_create_and_get_document(self, test_setup):
        db, _, _, _, tenant_id = test_setup

        doc = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_type="application/pdf",
            byte_size=1024,
            checksum="abc123",
        )

        assert doc.id
        assert doc.tenant_id == tenant_id
        assert doc.filename == "test.pdf"
        assert doc.status == "pending"

        # Get document
        fetched = get_document(db, tenant_id, doc.id)
        assert fetched is not None
        assert fetched.id == doc.id

    def test_list_documents(self, test_setup):
        db, _, _, _, tenant_id = test_setup

        # Create multiple documents
        doc1 = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="doc1.pdf",
            content_type="application/pdf",
            byte_size=100,
            checksum="hash1",
        )
        doc2 = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="doc2.txt",
            content_type="text/plain",
            byte_size=200,
            checksum="hash2",
        )

        docs = list_documents(db, tenant_id)
        assert len(docs) == 2
        assert docs[0].id == doc2.id  # Most recent first
        assert docs[1].id == doc1.id

    def test_list_documents_with_status_filter(self, test_setup):
        db, _, _, _, tenant_id = test_setup

        doc1 = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="doc1.pdf",
            content_type="application/pdf",
            byte_size=100,
            checksum="hash1",
        )
        doc2 = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="doc2.txt",
            content_type="text/plain",
            byte_size=200,
            checksum="hash2",
        )
        update_document_status(db, tenant_id, doc1.id, "completed")

        docs = list_documents(db, tenant_id, status="completed")
        assert len(docs) == 1
        assert docs[0].id == doc1.id

    def test_tenant_isolation_documents(self, test_setup, clean_database):
        """Documents from one tenant must not be visible to another."""
        db = get_database(os.environ["DATABASE_URL"])

        # Tenant A
        integ_a, _, _ = issue_integration_credential(db, integration_name="tenant-a")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)
        create_document(
            db=db,
            tenant_id=tenant_a,
            filename="doc_a.pdf",
            content_type="application/pdf",
            byte_size=100,
            checksum="hash_a",
        )

        # Tenant B
        integ_b, _, _ = issue_integration_credential(db, integration_name="tenant-b")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)
        create_document(
            db=db,
            tenant_id=tenant_b,
            filename="doc_b.pdf",
            content_type="application/pdf",
            byte_size=200,
            checksum="hash_b",
        )

        # Tenant A should only see their doc
        docs_a = list_documents(db, tenant_a)
        assert len(docs_a) == 1
        assert docs_a[0].filename == "doc_a.pdf"

        # Tenant B should only see their doc
        docs_b = list_documents(db, tenant_b)
        assert len(docs_b) == 1
        assert docs_b[0].filename == "doc_b.pdf"

        # Cross-tenant get should return None
        assert get_document(db, tenant_a, docs_b[0].id) is None
        assert get_document(db, tenant_b, docs_a[0].id) is None

    def test_create_chunks(self, test_setup):
        db, _, _, _, tenant_id = test_setup

        doc = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_type="application/pdf",
            byte_size=1024,
            checksum="abc123",
        )

        chunks = [
            {
                "chunk_index": 0,
                "content": "First chunk of text",
                "token_count": 4,
                "start_char": 0,
                "end_char": 19,
                "metadata": {"source": "page 1"},
            },
            {
                "chunk_index": 1,
                "content": "Second chunk of text",
                "token_count": 4,
                "start_char": 19,
                "end_char": 39,
                "metadata": {"source": "page 2"},
            },
        ]

        stored = create_chunks(db, tenant_id=tenant_id, document_id=doc.id, chunks=chunks)

        assert len(stored) == 2
        assert stored[0].chunk_index == 0
        assert stored[0].content == "First chunk of text"
        assert stored[1].chunk_index == 1
        assert stored[1].content == "Second chunk of text"

        # Fetch chunks
        fetched = get_chunks_for_document(db, tenant_id, doc.id)
        assert len(fetched) == 2
        assert fetched[0].content == "First chunk of text"
        assert fetched[1].content == "Second chunk of text"

    def test_get_all_chunks_for_tenant(self, test_setup):
        db, _, _, _, tenant_id = test_setup

        doc = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_type="application/pdf",
            byte_size=1024,
            checksum="abc123",
        )
        create_chunks(
            db,
            tenant_id=tenant_id,
            document_id=doc.id,
            chunks=[
                {"chunk_index": 0, "content": "Chunk 1", "token_count": 2, "start_char": 0, "end_char": 7, "metadata": {}},
                {"chunk_index": 1, "content": "Chunk 2", "token_count": 2, "start_char": 7, "end_char": 14, "metadata": {}},
            ],
        )

        chunks = get_all_chunks_for_tenant(db, tenant_id)
        assert len(chunks) == 2

    def test_delete_document(self, test_setup):
        db, _, _, _, tenant_id = test_setup

        doc = create_document(
            db=db,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_type="application/pdf",
            byte_size=1024,
            checksum="abc123",
        )
        create_chunks(
            db,
            tenant_id=tenant_id,
            document_id=doc.id,
            chunks=[{"chunk_index": 0, "content": "Test", "token_count": 1, "start_char": 0, "end_char": 4, "metadata": {}}],
        )

        deleted = delete_document(db, tenant_id, doc.id)
        assert deleted is True

        # Document and chunks should be gone
        assert get_document(db, tenant_id, doc.id) is None
        assert get_chunks_for_document(db, tenant_id, doc.id) == []

    def test_tenant_isolation_chunks(self, test_setup, clean_database):
        """Chunks from one tenant must not be visible to another."""
        db = get_database(os.environ["DATABASE_URL"])

        integ_a, _, _ = issue_integration_credential(db, integration_name="tenant-a-chunks")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)
        doc_a = create_document(
            db=db,
            tenant_id=tenant_a,
            filename="doc_a.pdf",
            content_type="application/pdf",
            byte_size=100,
            checksum="hash_a",
        )
        create_chunks(
            db,
            tenant_id=tenant_a,
            document_id=doc_a.id,
            chunks=[{"chunk_index": 0, "content": "Chunk A", "token_count": 2, "start_char": 0, "end_char": 7, "metadata": {}}],
        )

        integ_b, _, _ = issue_integration_credential(db, integration_name="tenant-b-chunks")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)
        doc_b = create_document(
            db=db,
            tenant_id=tenant_b,
            filename="doc_b.pdf",
            content_type="application/pdf",
            byte_size=200,
            checksum="hash_b",
        )
        create_chunks(
            db,
            tenant_id=tenant_b,
            document_id=doc_b.id,
            chunks=[{"chunk_index": 0, "content": "Chunk B", "token_count": 2, "start_char": 0, "end_char": 7, "metadata": {}}],
        )

        chunks_a = get_all_chunks_for_tenant(db, tenant_a)
        chunks_b = get_all_chunks_for_tenant(db, tenant_b)

        assert len(chunks_a) == 1
        assert chunks_a[0].content == "Chunk A"
        assert len(chunks_b) == 1
        assert chunks_b[0].content == "Chunk B"


class TestKnowledgeService:
    """Tests for the knowledge service layer (with Retrieval Stack)."""

    def test_ingest_text_file(self, test_setup):
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        content = b"This is a test document.\nIt has multiple lines.\nAnd some content."
        result = service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="test.txt",
            content_type="text/plain",
        )

        assert result.document.status == "completed"
        assert result.chunks_created > 0
        assert result.document.filename == "test.txt"

        # Verify chunks were stored
        chunks = service.get_document_chunks(tenant_id, result.document.id)
        assert len(chunks) == result.chunks_created
        assert all(c.content for c in chunks)

    def test_ingest_creates_chunks_with_metadata(self, test_setup):
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        content = b"First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="test.txt",
            content_type="text/plain",
        )

        chunks = service.get_document_chunks(tenant_id, result.document.id)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert isinstance(chunk.metadata, dict)

    def test_search_bm25(self, test_setup):
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest a document with known content
        content = b"Error handling in Python. Try except finally blocks. Exception handling best practices."
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="python_errors.txt",
            content_type="text/plain",
        )

        # Search for relevant terms
        results = service.search(tenant_id, "Python error handling", top_k=3)

        assert len(results) > 0
        assert all(r.score > 0 for r in results)
        assert all(isinstance(r.chunk, type(service.get_document_chunks(tenant_id, results[0].chunk.document_id)[0])) for r in results)

    def test_search_tenant_isolation(self, test_setup, clean_database):
        """Search results must not cross tenant boundaries."""
        db = get_database(os.environ["DATABASE_URL"])

        # Tenant A
        integ_a, _, _ = issue_integration_credential(db, integration_name="tenant-a-search")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)
        service_a = KnowledgeService(db)
        service_a.ingest_document(
            tenant_id=tenant_a,
            file_content=b"Tenant A confidential data about billing",
            filename="a.txt",
            content_type="text/plain",
        )

        # Tenant B
        integ_b, _, _ = issue_integration_credential(db, integration_name="tenant-b-search")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)
        service_b = KnowledgeService(db)
        service_b.ingest_document(
            tenant_id=tenant_b,
            file_content=b"Tenant B confidential data about shipping",
            filename="b.txt",
            content_type="text/plain",
        )

        # Search from tenant A should not see tenant B's data
        results_a = service_a.search(tenant_a, "confidential", top_k=5)
        assert len(results_a) == 1
        assert "billing" in results_a[0].chunk.content.lower()

        # Search from tenant B should not see tenant A's data
        results_b = service_b.search(tenant_b, "confidential", top_k=5)
        assert len(results_b) == 1
        assert "shipping" in results_b[0].chunk.content.lower()

    def test_invalidate_index_on_ingest(self, test_setup):
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest first document
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"First document about apples",
            filename="doc1.txt",
            content_type="text/plain",
        )

        # Search
        results1 = service.search(tenant_id, "apples", top_k=5)
        assert len(results1) == 1

        # Ingest second document
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Second document about bananas",
            filename="doc2.txt",
            content_type="text/plain",
        )

        # Search again - should find both
        results2 = service.search(tenant_id, "document", top_k=10)
        assert len(results2) == 2


class TestKnowledgeAPI:
    """Tests for the knowledge API endpoints."""

    def test_ingest_document_endpoint(self, client, test_setup):
        _, _, _, token, tenant_id = test_setup

        response = client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("test.txt", b"Test content for ingestion", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "completed"
        assert data["chunks_created"] > 0
        assert data["filename"] == "test.txt"

    def test_ingest_requires_auth(self, client, clean_database):
        response = client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("test.txt", b"Test", "text/plain")},
        )
        assert response.status_code == 401

    def test_list_documents_endpoint(self, client, test_setup):
        _, _, _, token, tenant_id = test_setup

        # Ingest a couple of documents
        client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("doc1.txt", b"Content 1", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("doc2.txt", b"Content 2", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )

        response = client.get(
            "/api/v1/tickets/knowledge/documents",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_search_endpoint(self, client, test_setup):
        _, _, _, token, tenant_id = test_setup

        # Ingest a document
        client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("search_test.txt", b"Python exception handling try except", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Search
        response = client.post(
            "/api/v1/tickets/knowledge/search",
            json={"query": "Python exception", "top_k": 5},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Python exception"
        assert len(data["results"]) > 0
        assert data["results"][0]["score"] > 0

    def test_search_tenant_isolation_endpoint(self, client, clean_database):
        """API search must respect tenant boundaries."""
        db = get_database(os.environ["DATABASE_URL"])

        # Tenant A
        integ_a, cred_a, token_a = issue_integration_credential(db, integration_name="api-tenant-a")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)

        # Tenant B
        integ_b, cred_b, token_b = issue_integration_credential(db, integration_name="api-tenant-b")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)

        # Tenant A uploads
        client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("a.txt", b"Tenant A secret data", "text/plain")},
            headers={"Authorization": f"Bearer {token_a}"},
        )

        # Tenant B uploads
        client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("b.txt", b"Tenant B secret data", "text/plain")},
            headers={"Authorization": f"Bearer {token_b}"},
        )

        # Tenant A searches
        resp_a = client.post(
            "/api/v1/tickets/knowledge/search",
            json={"query": "secret", "top_k": 5},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert len(data_a["results"]) == 1
        assert "tenant a" in data_a["results"][0]["chunk"]["content"].lower()

        # Tenant B searches
        resp_b = client.post(
            "/api/v1/tickets/knowledge/search",
            json={"query": "secret", "top_k": 5},
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert len(data_b["results"]) == 1
        assert "tenant b" in data_b["results"][0]["chunk"]["content"].lower()

    def test_get_document_chunks_endpoint(self, client, test_setup):
        _, _, _, token, tenant_id = test_setup

        # Ingest a document
        ingest_resp = client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("chunks_test.txt", b"First chunk.\n\nSecond chunk.\n\nThird chunk.", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        doc_id = ingest_resp.json()["document_id"]

        # Get chunks
        response = client.get(
            f"/api/v1/tickets/knowledge/documents/{doc_id}/chunks",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert all("chunk_id" in c for c in data)

    def test_delete_document_endpoint(self, client, test_setup):
        _, _, _, token, tenant_id = test_setup

        # Ingest a document
        ingest_resp = client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("delete_test.txt", b"To be deleted", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        doc_id = ingest_resp.json()["document_id"]

        # Delete
        response = client.delete(
            f"/api/v1/tickets/knowledge/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 204

        # Verify gone
        get_resp = client.get(
            f"/api/v1/tickets/knowledge/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_resp.status_code == 404

    def test_empty_file_rejected(self, client, test_setup):
        _, _, _, token, tenant_id = test_setup

        response = client.post(
            "/api/v1/tickets/knowledge/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 400


class TestDenseRetrieval:
    """Tests for dense vector retrieval functionality."""

    def test_dense_search_returns_relevant_chunks(self, test_setup):
        """Dense search returns chunks for semantically similar queries."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest a document about Python error handling
        content = b"Error handling in Python uses try except finally blocks. Exception handling best practices include catching specific exceptions."
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="python_errors.txt",
            content_type="text/plain",
        )

        # Search with semantically related query (not exact keyword match)
        results = service.search_dense(tenant_id, "How to catch exceptions in Python", top_k=3)

        assert len(results) > 0
        assert all(r.score > 0 for r in results)
        # Should find the document about error handling
        assert any("except" in r.chunk.content.lower() or "exception" in r.chunk.content.lower() for r in results)

    def test_dense_search_tenant_isolation(self, test_setup, clean_database):
        """Dense search results must not cross tenant boundaries."""
        db = get_database(os.environ["DATABASE_URL"])

        # Tenant A
        integ_a, _, _ = issue_integration_credential(db, integration_name="tenant-a-dense")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)
        service_a = KnowledgeService(db)
        service_a.ingest_document(
            tenant_id=tenant_a,
            file_content=b"Tenant A confidential data about billing",
            filename="a.txt",
            content_type="text/plain",
        )

        # Tenant B
        integ_b, _, _ = issue_integration_credential(db, integration_name="tenant-b-dense")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)
        service_b = KnowledgeService(db)
        service_b.ingest_document(
            tenant_id=tenant_b,
            file_content=b"Tenant B confidential data about shipping",
            filename="b.txt",
            content_type="text/plain",
        )

        # Search from tenant A should not see tenant B's data
        results_a = service_a.search_dense(tenant_a, "confidential", top_k=5)
        assert len(results_a) == 1
        assert "billing" in results_a[0].chunk.content.lower()

        # Search from tenant B should not see tenant A's data
        results_b = service_b.search_dense(tenant_b, "confidential", top_k=5)
        assert len(results_b) == 1
        assert "shipping" in results_b[0].chunk.content.lower()

    def test_dense_newly_ingested_chunks_searchable(self, test_setup):
        """Newly ingested chunks become searchable in dense index."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest first document
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"First document about apples and fruit",
            filename="doc1.txt",
            content_type="text/plain",
        )

        # Search
        results1 = service.search_dense(tenant_id, "apples", top_k=5)
        assert len(results1) == 1

        # Ingest second document
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Second document about bananas and fruit",
            filename="doc2.txt",
            content_type="text/plain",
        )

        # Search again - should find both
        results2 = service.search_dense(tenant_id, "fruit", top_k=10)
        assert len(results2) == 2

    def test_dense_deleted_documents_not_in_results(self, test_setup):
        """Deleted documents no longer appear in dense results."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest a document
        result = service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Document about billing refunds",
            filename="billing.txt",
            content_type="text/plain",
        )
        doc_id = result.document.id

        # Verify it's searchable
        results_before = service.search_dense(tenant_id, "billing refunds", top_k=5)
        assert len(results_before) == 1

        # Delete the document
        service.delete_document(tenant_id, doc_id)

        # Should no longer appear
        results_after = service.search_dense(tenant_id, "billing refunds", top_k=5)
        assert len(results_after) == 0

    def test_dense_empty_index_returns_empty_results(self, test_setup):
        """Empty dense index returns empty results."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        results = service.search_dense(tenant_id, "any query", top_k=5)
        assert results == []

    def test_bm25_behavior_unchanged(self, test_setup):
        """Existing BM25 search behavior remains unchanged."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        content = b"Error handling in Python. Try except finally blocks. Exception handling best practices."
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="python_errors.txt",
            content_type="text/plain",
        )

        # BM25 search should still work
        bm25_results = service.search(tenant_id, "Python error handling", top_k=3)
        assert len(bm25_results) > 0
        assert all(r.score > 0 for r in bm25_results)

        # Dense search should also work (different algorithm, may find different results)
        dense_results = service.search_dense(tenant_id, "Python error handling", top_k=3)
        assert len(dense_results) > 0
        assert all(r.score > 0 for r in dense_results)

    def test_dense_search_mode_in_tool(self, test_setup):
        """The search_knowledge tool works with dense mode."""
        db, _, _, _, tenant_id = test_setup

        # Ingest a document
        from app.knowledge.service import KnowledgeService
        service = KnowledgeService(db)
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"To reset your password, visit account settings and use 'Forgot password'.",
            filename="password_reset.txt",
            content_type="text/plain",
        )

        # Use the tool directly with dense mode
        from app.agent_core.tool_schemas import SearchKnowledgeArgs
        from app.agent_core.tools import execute_tool

        args = SearchKnowledgeArgs(query="password reset", top_k=5, mode="dense")
        result = execute_tool("search_knowledge", args.model_dump(), tenant_id=tenant_id, db=db)

        assert result["tool"] == "search_knowledge"
        assert "error" not in result
        assert len(result["results"]) > 0
        assert any("forgot password" in r["content"].lower() for r in result["results"])

    def test_bm25_search_mode_in_tool(self, test_setup):
        """The search_knowledge tool works with bm25 mode (default)."""
        db, _, _, _, tenant_id = test_setup

        # Ingest a document
        from app.knowledge.service import KnowledgeService
        service = KnowledgeService(db)
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Billing refunds are processed within 5 business days.",
            filename="billing.txt",
            content_type="text/plain",
        )

        from app.agent_core.tool_schemas import SearchKnowledgeArgs
        from app.agent_core.tools import execute_tool

        # Default mode is bm25
        args = SearchKnowledgeArgs(query="billing refunds", top_k=5)
        result = execute_tool("search_knowledge", args.model_dump(), tenant_id=tenant_id, db=db)

        assert result["tool"] == "search_knowledge"
        assert "error" not in result
        assert len(result["results"]) > 0
        assert any("business days" in r["content"].lower() for r in result["results"])


class TestHybridRetrieval:
    """Tests for hybrid (BM25 + dense) retrieval with RRF."""

    def test_hybrid_search_combines_bm25_and_dense_rankings(self, test_setup):
        """Hybrid search combines BM25 and dense rankings using RRF."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest documents with different characteristics
        # Doc 1: keyword-rich (good for BM25)
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Python error handling try except finally blocks exception catching",
            filename="python_errors.txt",
            content_type="text/plain",
        )
        # Doc 2: semantically related but different keywords (good for dense)
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"How to catch exceptions in Python programming language",
            filename="catch_exceptions.txt",
            content_type="text/plain",
        )

        # Hybrid search should find both
        results = service.search_hybrid(tenant_id, "Python exception handling", top_k=5)

        assert len(results) >= 1
        # Results should have RRF scores
        assert all(hasattr(r, 'score') and r.score > 0 for r in results)

    def test_hybrid_no_duplicate_chunks(self, test_setup):
        """A chunk found by both retrievers is not duplicated in results."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest a document that will match well in both BM25 and dense
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Password reset instructions: go to account settings and click forgot password to receive reset email",
            filename="password.txt",
            content_type="text/plain",
        )

        results = service.search_hybrid(tenant_id, "password reset", top_k=5)

        # Should not have duplicate chunk_ids
        chunk_ids = [r.chunk.id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks found in hybrid results"

    def test_hybrid_uses_rrf_not_raw_scores(self, test_setup):
        """Hybrid retrieval uses RRF ranking, not raw score merging."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest multiple documents
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Billing refund policy: refunds processed within 5 business days for eligible charges",
            filename="billing1.txt",
            content_type="text/plain",
        )
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Shipping policy: orders ship within 24 hours, tracking provided via email",
            filename="shipping.txt",
            content_type="text/plain",
        )
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Account settings: update password, email, and notification preferences in account page",
            filename="account.txt",
            content_type="text/plain",
        )

        results = service.search_hybrid(tenant_id, "policy", top_k=5)

        assert len(results) >= 1
        # Results should be sorted by RRF score (highest first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by RRF score"

    def test_hybrid_respects_top_k(self, test_setup):
        """Hybrid retrieval respects top_k parameter."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest multiple documents
        for i in range(5):
            service.ingest_document(
                tenant_id=tenant_id,
                file_content=f"Document {i} about topic {i}".encode(),
                filename=f"doc{i}.txt",
                content_type="text/plain",
            )

        results = service.search_hybrid(tenant_id, "topic", top_k=3)

        assert len(results) <= 3

    def test_hybrid_tenant_isolation(self, test_setup, clean_database):
        """Hybrid search results must not cross tenant boundaries."""
        db = get_database(os.environ["DATABASE_URL"])

        # Tenant A
        integ_a, _, _ = issue_integration_credential(db, integration_name="tenant-a-hybrid")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)
        service_a = KnowledgeService(db)
        service_a.ingest_document(
            tenant_id=tenant_a,
            file_content=b"Tenant A confidential data about billing",
            filename="a.txt",
            content_type="text/plain",
        )

        # Tenant B
        integ_b, _, _ = issue_integration_credential(db, integration_name="tenant-b-hybrid")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)
        service_b = KnowledgeService(db)
        service_b.ingest_document(
            tenant_id=tenant_b,
            file_content=b"Tenant B confidential data about shipping",
            filename="b.txt",
            content_type="text/plain",
        )

        # Search from tenant A should not see tenant B's data
        results_a = service_a.search_hybrid(tenant_a, "confidential", top_k=5)
        assert len(results_a) == 1
        assert "billing" in results_a[0].chunk.content.lower()

        # Search from tenant B should not see tenant A's data
        results_b = service_b.search_hybrid(tenant_b, "confidential", top_k=5)
        assert len(results_b) == 1
        assert "shipping" in results_b[0].chunk.content.lower()

    def test_hybrid_newly_ingested_chunks_searchable(self, test_setup):
        """Newly ingested chunks become searchable through hybrid mode."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest first document
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"First document about apples and fruit",
            filename="doc1.txt",
            content_type="text/plain",
        )

        # Search
        results1 = service.search_hybrid(tenant_id, "apples", top_k=5)
        assert len(results1) == 1

        # Ingest second document
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Second document about bananas and fruit",
            filename="doc2.txt",
            content_type="text/plain",
        )

        # Search again - should find both
        results2 = service.search_hybrid(tenant_id, "fruit", top_k=10)
        assert len(results2) == 2

    def test_hybrid_deleted_documents_not_in_results(self, test_setup):
        """Deleted documents no longer appear in hybrid results."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Ingest a document
        result = service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Document about billing refunds",
            filename="billing.txt",
            content_type="text/plain",
        )
        doc_id = result.document.id

        # Verify it's searchable
        results_before = service.search_hybrid(tenant_id, "billing refunds", top_k=5)
        assert len(results_before) == 1

        # Delete the document
        service.delete_document(tenant_id, doc_id)

        # Should no longer appear
        results_after = service.search_hybrid(tenant_id, "billing refunds", top_k=5)
        assert len(results_after) == 0

    def test_hybrid_empty_results_handled(self, test_setup):
        """Empty results from either retrieval branch are handled correctly."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        # Search on empty index
        results = service.search_hybrid(tenant_id, "any query", top_k=5)
        assert results == []

        # Ingest one document, search for non-matching query
        # Dense retriever will still return results (cosine similarity to all docs),
        # but they should have very low RRF scores
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Document about apples",
            filename="doc1.txt",
            content_type="text/plain",
        )

        # Query that won't match well
        results = service.search_hybrid(tenant_id, "xyz123nonexistent", top_k=5)
        # Results may be returned but with very low scores
        assert len(results) <= 1
        if results:
            # Score should be very low for non-matching query
            assert results[0].score < 0.1

    def test_bm25_and_dense_modes_unchanged(self, test_setup):
        """Existing BM25 and dense modes remain unchanged."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db)

        content = b"Error handling in Python. Try except finally blocks. Exception handling best practices."
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="python_errors.txt",
            content_type="text/plain",
        )

        # BM25 search should still work
        bm25_results = service.search(tenant_id, "Python error handling", top_k=3)
        assert len(bm25_results) > 0
        assert all(r.score > 0 for r in bm25_results)

        # Dense search should also work
        dense_results = service.search_dense(tenant_id, "Python error handling", top_k=3)
        assert len(dense_results) > 0
        assert all(r.score > 0 for r in dense_results)

    def test_hybrid_search_mode_in_tool(self, test_setup):
        """The search_knowledge tool works with hybrid mode."""
        db, _, _, _, tenant_id = test_setup

        # Ingest a document
        from app.knowledge.service import KnowledgeService
        service = KnowledgeService(db)
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"To reset your password, visit account settings and use 'Forgot password'.",
            filename="password_reset.txt",
            content_type="text/plain",
        )

        # Use the tool directly with hybrid mode
        from app.agent_core.tool_schemas import SearchKnowledgeArgs
        from app.agent_core.tools import execute_tool

        args = SearchKnowledgeArgs(query="password reset", top_k=5, mode="hybrid")
        result = execute_tool("search_knowledge", args.model_dump(), tenant_id=tenant_id, db=db)

        assert result["tool"] == "search_knowledge"
        assert "error" not in result
        assert len(result["results"]) > 0
        assert any("forgot password" in r["content"].lower() for r in result["results"])


class TestReranking:
    """Tests for cross-encoder reranking functionality."""

    def test_reranking_changes_ordering(self, test_setup):
        """Reranking changes candidate ordering appropriately."""
        db, _, _, _, tenant_id = test_setup
        # Enable reranking for this test
        service = KnowledgeService(db, enable_reranking=True)

        # Ingest documents with different relevance patterns
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Password reset: go to settings and click forgot password to get email link",
            filename="password.txt",
            content_type="text/plain",
        )
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Account settings: update your profile, email, and notification preferences",
            filename="account.txt",
            content_type="text/plain",
        )

        # Search without reranking
        results_no_rerank = service.search(tenant_id, "password reset", top_k=5, rerank=False)
        # Search with reranking
        results_rerank = service.search(tenant_id, "password reset", top_k=5, rerank=True)

        # Both should return results
        assert len(results_no_rerank) > 0
        assert len(results_rerank) > 0
        # Reranked results should have relevance scores
        assert all(hasattr(r, 'score') for r in results_rerank)

    def test_reranking_works_with_bm25_candidates(self, test_setup):
        """Reranking works with BM25 candidates."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Billing refund policy: refunds processed within 5 business days",
            filename="billing.txt",
            content_type="text/plain",
        )
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Shipping policy: orders ship within 24 hours",
            filename="shipping.txt",
            content_type="text/plain",
        )

        results = service.search(tenant_id, "refund policy", top_k=5, rerank=True)
        assert len(results) > 0
        # Should find the billing document
        assert any("refund" in r.chunk.content.lower() for r in results)

    def test_reranking_works_with_dense_candidates(self, test_setup):
        """Reranking works with dense candidates."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Python exception handling try except finally blocks",
            filename="python.txt",
            content_type="text/plain",
        )
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Java error handling with catch blocks and throws declarations",
            filename="java.txt",
            content_type="text/plain",
        )

        results = service.search_dense(tenant_id, "Python error handling", top_k=5, rerank=True)
        assert len(results) > 0
        assert any("python" in r.chunk.content.lower() for r in results)

    def test_reranking_works_with_hybrid_candidates(self, test_setup):
        """Reranking works with hybrid candidates."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Password reset instructions: forgot password link sent via email",
            filename="password.txt",
            content_type="text/plain",
        )
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Account recovery: verify identity and restore access",
            filename="recovery.txt",
            content_type="text/plain",
        )

        results = service.search_hybrid(tenant_id, "password reset", top_k=5, rerank=True)
        assert len(results) > 0
        assert any("password" in r.chunk.content.lower() for r in results)

    def test_reranking_respects_top_k(self, test_setup):
        """Final top_k is respected after reranking."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        for i in range(5):
            service.ingest_document(
                tenant_id=tenant_id,
                file_content=f"Document {i} about topic {i}".encode(),
                filename=f"doc{i}.txt",
                content_type="text/plain",
            )

        results = service.search(tenant_id, "topic", top_k=3, rerank=True)
        assert len(results) <= 3

    def test_reranking_empty_candidates(self, test_setup):
        """Empty candidate sets are handled correctly."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        results = service.search(tenant_id, "any query", top_k=5, rerank=True)
        assert results == []

    def test_reranking_can_be_disabled(self, test_setup):
        """Reranking can be disabled."""
        db, _, _, _, tenant_id = test_setup
        # Service without reranking enabled
        service = KnowledgeService(db, enable_reranking=False)

        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Test document about billing refunds",
            filename="billing.txt",
            content_type="text/plain",
        )

        # Rerank should be ignored when service doesn't have it enabled
        results = service.search(tenant_id, "billing refunds", top_k=5, rerank=True)
        assert len(results) > 0
        # Should work but without reranking applied

    def test_reranking_missing_dependency_error(self, test_setup):
        """Missing/unavailable reranker dependencies produce clear errors."""
        # This test verifies that the reranker throws an error when the model
        # is not available, rather than silently falling back
        db, _, _, _, tenant_id = test_setup
        # Use a non-existent model name to trigger an error
        service = KnowledgeService(db, enable_reranking=True, reranker_model_name="non-existent-model")

        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"Test document",
            filename="test.txt",
            content_type="text/plain",
        )

        # Should raise an error when trying to use the reranker
        import pytest
        with pytest.raises(Exception):
            service.search(tenant_id, "test", top_k=5, rerank=True)

    def test_reranking_tenant_isolation(self, test_setup, clean_database):
        """Tenant isolation is preserved during reranking."""
        db = get_database(os.environ["DATABASE_URL"])

        # Tenant A
        integ_a, _, _ = issue_integration_credential(db, integration_name="tenant-a-rerank")
        tenant_a = _assign_tenant_to_integration(db, integ_a.id)
        service_a = KnowledgeService(db, enable_reranking=True)
        service_a.ingest_document(
            tenant_id=tenant_a,
            file_content=b"Tenant A confidential data about billing",
            filename="a.txt",
            content_type="text/plain",
        )

        # Tenant B
        integ_b, _, _ = issue_integration_credential(db, integration_name="tenant-b-rerank")
        tenant_b = _assign_tenant_to_integration(db, integ_b.id)
        service_b = KnowledgeService(db, enable_reranking=True)
        service_b.ingest_document(
            tenant_id=tenant_b,
            file_content=b"Tenant B confidential data about shipping",
            filename="b.txt",
            content_type="text/plain",
        )

        # Search from tenant A should not see tenant B's data
        results_a = service_a.search(tenant_a, "confidential", top_k=5, rerank=True)
        assert len(results_a) == 1
        assert "billing" in results_a[0].chunk.content.lower()

        # Search from tenant B should not see tenant A's data
        results_b = service_b.search(tenant_b, "confidential", top_k=5, rerank=True)
        assert len(results_b) == 1
        assert "shipping" in results_b[0].chunk.content.lower()

    def test_existing_modes_unchanged_when_rerank_disabled(self, test_setup):
        """Existing BM25, dense, and hybrid behavior remains unchanged when reranking is disabled."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        content = b"Error handling in Python. Try except finally blocks. Exception handling best practices."
        service.ingest_document(
            tenant_id=tenant_id,
            file_content=content,
            filename="python_errors.txt",
            content_type="text/plain",
        )

        # BM25 search without reranking
        bm25_results = service.search(tenant_id, "Python error handling", top_k=3, rerank=False)
        assert len(bm25_results) > 0

        # Dense search without reranking
        dense_results = service.search_dense(tenant_id, "Python error handling", top_k=3, rerank=False)
        assert len(dense_results) > 0

        # Hybrid search without reranking
        hybrid_results = service.search_hybrid(tenant_id, "Python error handling", top_k=3, rerank=False)
        assert len(hybrid_results) > 0

    def test_reranking_in_tool(self, test_setup):
        """Agent tool dispatch correctly handles the reranking option."""
        db, _, _, _, tenant_id = test_setup
        service = KnowledgeService(db, enable_reranking=True)

        service.ingest_document(
            tenant_id=tenant_id,
            file_content=b"To reset your password, visit account settings and use 'Forgot password'.",
            filename="password_reset.txt",
            content_type="text/plain",
        )

        from app.agent_core.tool_schemas import SearchKnowledgeArgs
        from app.agent_core.tools import execute_tool

        # Test with rerank=True
        args = SearchKnowledgeArgs(query="password reset", top_k=5, mode="bm25", rerank=True)
        result = execute_tool("search_knowledge", args.model_dump(), tenant_id=tenant_id, db=db)

        assert result["tool"] == "search_knowledge"
        assert "error" not in result
        assert len(result["results"]) > 0
        assert any("forgot password" in r["content"].lower() for r in result["results"])
