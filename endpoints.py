from fastapi import APIRouter, HTTPException
from rag import get_wikipedia_rag_response

router = APIRouter()

@router.get("/query/")
async def query_rag_system(query: str):
    try:
        response = await get_wikipedia_rag_response(query)
        # Ensure response is always a string
        return {"query": query, "response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))