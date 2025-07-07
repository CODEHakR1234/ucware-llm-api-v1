
from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
from app.cache.cache_db import get_cache_db

router = APIRouter(prefix="/cache", tags=["cache-management"])

@router.get("/statistics")
async def get_cache_statistics(cache = Depends(get_cache_db)):
    """캐시 통계 정보 조회"""
    return cache.get_statistics()

@router.get("/summaries/{date}")
async def get_summaries_by_date(
    date: str,
    cache = Depends(get_cache_db)
):
    """특정 날짜의 모든 요약본 조회"""
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        summaries = cache.get_summaries_by_date(date_obj)
        return {
            "date": date,
            "count": len(summaries),
            "file_ids": list(summaries.keys())
        }
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

@router.delete("/cleanup")
async def cleanup_expired(cache = Depends(get_cache_db)):
    """만료된 요약본 수동 정리"""
    cache.cleanup_expired_summaries()
    return {"message": "Cleanup completed"}

@router.delete("/summary/{file_id}")
async def delete_specific_summary(
    file_id: str,
    cache = Depends(get_cache_db)
):
    """특정 요약본 삭제"""
    success = cache.delete_pdf(file_id)
    return {
        "file_id": file_id,
        "deleted": success
    }