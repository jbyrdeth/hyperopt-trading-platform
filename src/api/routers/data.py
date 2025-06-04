"""
Data Router (Placeholder)

TODO: Implement data management endpoints for Task 13.6
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def data_placeholder():
    """Data management endpoints coming in Task 13.6"""
    return {"message": "Data management endpoints coming in Task 13.6"} 