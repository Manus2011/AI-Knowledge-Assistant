from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=8, max_length=128)


@router.post("/register", status_code=201)
def register(creds: Credentials):
    try:
        auth_service.register(creds.username, creds.password)
    except auth_service.AuthError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"username": creds.username}


@router.post("/login")
def login(creds: Credentials):
    try:
        token = auth_service.authenticate(creds.username, creds.password)
    except auth_service.AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"access_token": token, "token_type": "bearer"}
