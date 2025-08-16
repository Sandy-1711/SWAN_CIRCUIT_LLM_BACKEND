from fastapi import APIRouter, HTTPException, status

router = APIRouter()

@router.get("/chat")
async def chat_endpoint():
    return {"message": "Chat endpoint is under construction"}
@router.get("/chat/health")
async def chat_health_check():
    return {"status": "healthy"}


