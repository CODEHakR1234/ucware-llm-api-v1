# app/background/cleanup_scheduler.py

import asyncio
from datetime import datetime, timedelta

from app.vectordb.vector_db import get_vector_db
from app.cache.cache_db import get_cache_db


def register_cleanup_task():
    async def cleanup_job():
        # ✅ (1) 서버 시작 시 1회 정리
        vdb = get_vector_db()
        cache = get_cache_db()
        deleted = vdb.cleanup_unused_vectors(cache)
        if deleted:
            print(f"[Startup Cleanup] Deleted {len(deleted)} vector(s): {deleted}")
        else:
            print(f"[Startup Cleanup] No vector deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ✅ (2) 다음 새벽 3시까지 대기
        while True:
            now = datetime.now()
            next_run = (now + timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
            if now.hour < 3:
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)

            delay = (next_run - now).total_seconds()
            print(f"[Auto Cleanup] Waiting {delay / 3600:.2f} hours until next cleanup at 03:00")

            await asyncio.sleep(delay)

            # ✅ (3) 매일 03:00 정리 실행
            vdb = get_vector_db()
            cache = get_cache_db()
            deleted = vdb.cleanup_unused_vectors(cache)
            if deleted:
                print(f"[Auto Cleanup] Deleted {len(deleted)} vector(s): {deleted}")
            else:
                print(f"[Auto Cleanup] No vector deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            await asyncio.sleep(86400)  # 그 다음 정리는 24시간 후

    asyncio.create_task(cleanup_job())
