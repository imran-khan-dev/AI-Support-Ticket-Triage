"""Knowledge base repository layer.

All operations are tenant-scoped: every query includes the tenant_id so that
one tenant can never read or write another tenant's documents or chunks.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from app.database import Database
from app.database.repository import _canonical_uuid


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    tenant_id: str
    filename: str
    content_type: str
    byte_size: int
    checksum: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: str
    updated_at: str
    completed_at: str | None
    error_message: str | None


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    tenant_id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    metadata: dict
    created_at: str


def _document_from_row(row) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        filename=row["filename"],
        content_type=row["content_type"],
        byte_size=row["byte_size"],
        checksum=row["checksum"],
        status=row["status"],
        created_at=row["created_at"].isoformat() + "Z",
        updated_at=row["updated_at"].isoformat() + "Z",
        completed_at=row["completed_at"].isoformat() + "Z" if row["completed_at"] else None,
        error_message=row["error_message"],
    )


def _chunk_from_row(row) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        document_id=str(row["document_id"]),
        chunk_index=row["chunk_index"],
        content=row["content"],
        token_count=row["token_count"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"].isoformat() + "Z",
    )


def create_document(
    db: Database,
    *,
    tenant_id: str,
    filename: str,
    content_type: str,
    byte_size: int,
    checksum: str,
) -> KnowledgeDocument:
    """Create a new knowledge document record with status 'pending'."""
    tenant_canonical = _canonical_uuid(tenant_id)
    if tenant_canonical is None:
        raise ValueError("tenant_id is not a valid UUID")

    with db.write_transaction() as conn:
        row = conn.execute(
            """
            INSERT INTO knowledge_documents
                (id, tenant_id, filename, content_type, byte_size, checksum, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            RETURNING id, tenant_id, filename, content_type, byte_size, checksum,
                      status, created_at, updated_at, completed_at, error_message
            """,
            (
                str(uuid.uuid4()),
                tenant_canonical,
                filename,
                content_type,
                byte_size,
                checksum,
            ),
        ).fetchone()
    if row is None:
        raise RuntimeError("The document was not stored.")
    return _document_from_row(row)


def get_document(db: Database, tenant_id: str, document_id: str) -> KnowledgeDocument | None:
    """Get a document by ID, scoped to tenant."""
    tenant_canonical = _canonical_uuid(tenant_id)
    doc_canonical = _canonical_uuid(document_id)
    if tenant_canonical is None or doc_canonical is None:
        return None

    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT id, tenant_id, filename, content_type, byte_size, checksum,
                   status, created_at, updated_at, completed_at, error_message
            FROM knowledge_documents
            WHERE id = %s AND tenant_id = %s
            """,
            (doc_canonical, tenant_canonical),
        ).fetchone()
    if row is None:
        return None
    return _document_from_row(row)


def list_documents(
    db: Database,
    tenant_id: str,
    *,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[KnowledgeDocument]:
    """List documents for a tenant, optionally filtered by status."""
    tenant_canonical = _canonical_uuid(tenant_id)
    if tenant_canonical is None:
        return []

    with db.connection() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT id, tenant_id, filename, content_type, byte_size, checksum,
                       status, created_at, updated_at, completed_at, error_message
                FROM knowledge_documents
                WHERE tenant_id = %s AND status = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (tenant_canonical, status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, tenant_id, filename, content_type, byte_size, checksum,
                       status, created_at, updated_at, completed_at, error_message
                FROM knowledge_documents
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (tenant_canonical, limit, offset),
            ).fetchall()
    return [_document_from_row(row) for row in rows]


def update_document_status(
    db: Database,
    tenant_id: str,
    document_id: str,
    status: Literal["pending", "processing", "completed", "failed"],
    *,
    error_message: str | None = None,
) -> KnowledgeDocument | None:
    """Update document status and optionally set error message."""
    tenant_canonical = _canonical_uuid(tenant_id)
    doc_canonical = _canonical_uuid(document_id)
    if tenant_canonical is None or doc_canonical is None:
        return None

    completed_at = "now()" if status == "completed" else "NULL"

    with db.write_transaction() as conn:
        row = conn.execute(
            f"""
            UPDATE knowledge_documents
            SET status = %s,
                updated_at = now(),
                completed_at = {completed_at},
                error_message = %s
            WHERE id = %s AND tenant_id = %s
            RETURNING id, tenant_id, filename, content_type, byte_size, checksum,
                      status, created_at, updated_at, completed_at, error_message
            """,
            (status, error_message, doc_canonical, tenant_canonical),
        ).fetchone()
    if row is None:
        return None
    return _document_from_row(row)


def create_chunks(
    db: Database,
    *,
    tenant_id: str,
    document_id: str,
    chunks: list[dict],
) -> list[KnowledgeChunk]:
    """Bulk insert chunks for a document."""
    tenant_canonical = _canonical_uuid(tenant_id)
    doc_canonical = _canonical_uuid(document_id)
    if tenant_canonical is None or doc_canonical is None:
        raise ValueError("Invalid UUID for tenant_id or document_id")

    if not chunks:
        return []

    with db.write_transaction() as conn:
        inserted = []
        for chunk in chunks:
            row = conn.execute(
                """
                INSERT INTO knowledge_chunks
                    (id, tenant_id, document_id, chunk_index, content,
                     token_count, start_char, end_char, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, tenant_id, document_id, chunk_index, content,
                          token_count, start_char, end_char, metadata, created_at
                """,
                (
                    str(uuid.uuid4()),
                    tenant_canonical,
                    doc_canonical,
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk["token_count"],
                    chunk["start_char"],
                    chunk["end_char"],
                    json.dumps(chunk.get("metadata", {})),
                ),
            ).fetchone()
            if row is not None:
                inserted.append(_chunk_from_row(row))
    return inserted


def get_chunks_for_document(
    db: Database, tenant_id: str, document_id: str
) -> list[KnowledgeChunk]:
    """Get all chunks for a document, scoped to tenant."""
    tenant_canonical = _canonical_uuid(tenant_id)
    doc_canonical = _canonical_uuid(document_id)
    if tenant_canonical is None or doc_canonical is None:
        return []

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tenant_id, document_id, chunk_index, content,
                   token_count, start_char, end_char, metadata, created_at
            FROM knowledge_chunks
            WHERE document_id = %s AND tenant_id = %s
            ORDER BY chunk_index
            """,
            (doc_canonical, tenant_canonical),
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def get_all_chunks_for_tenant(db: Database, tenant_id: str) -> list[KnowledgeChunk]:
    """Get all chunks for a tenant (for building the BM25 index)."""
    tenant_canonical = _canonical_uuid(tenant_id)
    if tenant_canonical is None:
        return []

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tenant_id, document_id, chunk_index, content,
                   token_count, start_char, end_char, metadata, created_at
            FROM knowledge_chunks
            WHERE tenant_id = %s
            ORDER BY document_id, chunk_index
            """,
            (tenant_canonical,),
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def delete_document(db: Database, tenant_id: str, document_id: str) -> bool:
    """Delete a document and its chunks (cascades). Returns True if deleted."""
    tenant_canonical = _canonical_uuid(tenant_id)
    doc_canonical = _canonical_uuid(document_id)
    if tenant_canonical is None or doc_canonical is None:
        return False

    with db.write_transaction() as conn:
        result = conn.execute(
            "DELETE FROM knowledge_documents WHERE id = %s AND tenant_id = %s",
            (doc_canonical, tenant_canonical),
        )
    return result.rowcount > 0