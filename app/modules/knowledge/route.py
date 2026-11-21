"""Knowledge base API routes.

All endpoints require integration authentication and are tenant-scoped.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.database import get_database
from app.modules.integrations.auth import require_integration
from app.knowledge.service import KnowledgeService

from .schemas import (
    ChunkResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)

router = APIRouter(tags=["knowledge"])


def get_knowledge_service() -> KnowledgeService:
    db = get_database()
    return KnowledgeService(db)


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def ingest_document(
    file: UploadFile = File(...),
    service: KnowledgeService = Depends(get_knowledge_service),
    integration=Depends(require_integration),
):
    tenant_id = integration.tenant_id

    file_content = await file.read()
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    filename = file.filename or "unknown"
    content_type = file.content_type or "application/octet-stream"

    try:
        result = service.ingest_document(
            tenant_id=tenant_id,
            file_content=file_content,
            filename=filename,
            content_type=content_type,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )

    return DocumentIngestResponse(
        document_id=result.document.id,
        filename=result.document.filename,
        content_type=result.document.content_type,
        byte_size=result.document.byte_size,
        checksum=result.document.checksum,
        status=result.document.status,
        chunks_created=result.chunks_created,
        created_at=result.document.created_at,
    )


@router.get(
    "/documents",
    response_model=list[DocumentListResponse],
    summary="List documents for the tenant",
)
async def list_documents(
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
    service: KnowledgeService = Depends(get_knowledge_service),
    integration=Depends(require_integration),
):
    tenant_id = integration.tenant_id

    docs = service.list_documents(
        tenant_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    return [
        DocumentListResponse(
            document_id=d.id,
            filename=d.filename,
            content_type=d.content_type,
            byte_size=d.byte_size,
            checksum=d.checksum,
            status=d.status,
            created_at=d.created_at,
            updated_at=d.updated_at,
            completed_at=d.completed_at,
            error_message=d.error_message,
        )
        for d in docs
    ]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentListResponse,
    summary="Get a document by ID",
)
async def get_document(
    document_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
    integration=Depends(require_integration),
):
    tenant_id = integration.tenant_id

    doc = service.get_document(tenant_id, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentListResponse(
        document_id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        byte_size=doc.byte_size,
        checksum=doc.checksum,
        status=doc.status,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        completed_at=doc.completed_at,
        error_message=doc.error_message,
    )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkResponse],
    summary="Get chunks for a document",
)
async def get_document_chunks(
    document_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
    integration=Depends(require_integration),
):
    tenant_id = integration.tenant_id

    doc = service.get_document(tenant_id, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    chunks = service.get_document_chunks(tenant_id, document_id)

    return [
        ChunkResponse(
            chunk_id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            token_count=c.token_count,
            start_char=c.start_char,
            end_char=c.end_char,
            metadata=c.metadata,
            created_at=c.created_at,
        )
        for c in chunks
    ]


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search knowledge base using BM25",
)
async def search_knowledge(
    request: SearchRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
    integration=Depends(require_integration),
):
    tenant_id = integration.tenant_id

    results = service.search(tenant_id, request.query, request.top_k)

    return SearchResponse(
        results=[
            SearchResultResponse(
                chunk=ChunkResponse(
                    chunk_id=r.chunk.id,
                    document_id=r.chunk.document_id,
                    chunk_index=r.chunk.chunk_index,
                    content=r.chunk.content,
                    token_count=r.chunk.token_count,
                    start_char=r.chunk.start_char,
                    end_char=r.chunk.end_char,
                    metadata=r.chunk.metadata,
                    created_at=r.chunk.created_at,
                ),
                score=r.score,
            )
            for r in results
        ],
        query=request.query,
        top_k=request.top_k,
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(
    document_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
    integration=Depends(require_integration),
):
    tenant_id = integration.tenant_id

    deleted = service.delete_document(tenant_id, document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)