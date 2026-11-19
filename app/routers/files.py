"""Multipart upload and deletion of conversation files."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.db import get_conversation_for_user, get_db
from app.models import Chunk, File as StoredFile, utcnow
from app.schemas import FileOut
from app.services.chunker import chunk_pages
from app.services.parser import MAX_FILES_PER_REQUEST, ParseError, parse_upload

router = APIRouter(tags=["files"])


@router.post("/conversations/{conversation_id}/files", response_model=list[FileOut])
async def upload_files(
    conversation_id: uuid.UUID,
    files: list[UploadFile],
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> list[StoredFile]:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload at most 5 files at a time",
        )

    conversation = await get_conversation_for_user(session, conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    parsed_files = []
    for upload in files:
        data = await upload.read()
        try:
            parsed_files.append(parse_upload(upload.filename or "", data))
        except ParseError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    created: list[StoredFile] = []
    for parsed in parsed_files:
        record = StoredFile(
            conversation_id=conversation.id,
            user_id=user_id,
            name=parsed.name,
            mime=parsed.mime,
            size=parsed.size,
            extracted_text=parsed.extracted_text,
            image_b64=parsed.image_b64,
        )
        session.add(record)
        await session.flush()
        for idx, text, page in chunk_pages(parsed.pages):
            session.add(Chunk(file_id=record.id, idx=idx, text=text, page=page))
        created.append(record)

    conversation.updated_at = utcnow()
    await session.flush()
    return created


@router.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: uuid.UUID,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    result = await session.execute(
        select(StoredFile).where(StoredFile.id == file_id, StoredFile.user_id == user_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    await session.delete(record)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
