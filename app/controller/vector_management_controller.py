
from fastapi import APIRouter
from datetime import datetime
from fastapi import Query
from app.vectordb.vector_db import get_vector_db
from app.cache.cache_db import get_cache_db  # ⬅️ 캐시 연동 추가

router = APIRouter(prefix="/vector", tags=["vector-management"])

@router.get("/list")
async def list_documents(vdb = get_vector_db()):
    return {"file_ids": vdb.list_stored_documents()}

@router.get("/statistics")
async def vector_statistics(vdb = get_vector_db()):
    file_ids = vdb.list_stored_documents()
    return {
        "total_collections": len(file_ids),
        "file_ids": file_ids
    }

@router.get("/check/{file_id}")
async def check_vector_exists(file_id: str, vdb = get_vector_db()):
    return {
        "file_id": file_id,
        "exists": file_id in vdb.list_stored_documents()
    }

@router.delete("/cleanup-unused")
async def cleanup_unused_vectors(vdb = get_vector_db(), cache = get_cache_db()):
    """Redis 캐시에 없는 오래된 vector 컬렉션 삭제"""
    deleted = vdb.cleanup_unused_vectors(cache)
    return {
        "deleted_count": len(deleted),
        "deleted_file_ids": deleted
    }

@router.get("/cleanup-log")
async def get_cleanup_log(
    date: str = Query(..., description="YYYY-MM-DD 형식의 날짜"),
    cache = get_cache_db()
):
    key = f"vector:deleted:{date}"
    logs = cache.r.lrange(key, 0, -1)
    return {
        "date": date,
        "deleted_file_ids": [entry.split("|")[0] for entry in logs],
        "raw_entries": logs
    }

