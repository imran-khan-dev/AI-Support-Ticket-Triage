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