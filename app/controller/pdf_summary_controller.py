from fastapi import APIRouter, Depends, HTTPException
from app.model.summary_dto import SummaryRequestDTO
from app.receiver.pdf_receiver import PDFReceiver
from app.vectordb.vector_db import get_vector_db
from app.cache.cache_db import get_cache_db
from app.service.summary_service import SummaryService

router = APIRouter()

@router.post("/summary")
async def summarize_pdf(
    req: SummaryRequestDTO,
    vdb = Depends(get_vector_db),
    cache = Depends(get_cache_db),
):
    if (c := cache.get_pdf(req.file_id)):
        return {"file_id": req.file_id, "summary": c, "cached": True}

    text = PDFReceiver().fetch_and_extract_text(req.pdf_url)
    if not text.strip():
        raise HTTPException(400, "PDF 텍스트 추출 실패")
    vdb.store(text, req.file_id)
    summary = SummaryService(vdb, cache).generate(req.file_id)
    return {"file_id": req.file_id, "summary": summary, "cached": False}

