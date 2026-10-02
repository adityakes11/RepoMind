import hashlib

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from backend.dependencies import bearer_scheme, get_current_user
from backend.schemas.auth import AuthRequest, AuthResponse, UserResponse
from backend.services.auth_service import authenticate, hash_password, issue_session, normalize_email, user_response
from src.database import create_user, delete_session, get_user_by_email


router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(request: AuthRequest):
    try:
        email = normalize_email(request.email)
        if get_user_by_email(email):
            raise HTTPException(status_code=409, detail="An account with this email already exists")
        user_id = create_user(email, hash_password(request.password))
        user = {"id": user_id, "email": email, "created_at": None}
        stored_user = get_user_by_email(email)
        token, expires_at = issue_session(stored_user)
        return {"access_token": token, "expires_at": expires_at, "user": user_response(stored_user)}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/login", response_model=AuthResponse)
def login(request: AuthRequest):
    try:
        user = authenticate(request.email, request.password)
        token, expires_at = issue_session(user)
        return {"access_token": token, "expires_at": expires_at, "user": user_response(user)}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me", response_model=UserResponse)
def me(user=Depends(get_current_user)):
    return user_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials and credentials.scheme.lower() == "bearer":
        delete_session(hashlib.sha256(credentials.credentials.encode("utf-8")).hexdigest())