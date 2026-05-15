from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_auth_service, get_current_user, get_optional_access_token
from db.models import UserAccount
from services.auth_service import AuthService, AuthenticationError, AuthenticatedSession

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthUserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None = None
    tenant_id: str


class AuthSessionResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserResponse


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User's email address. Must be unique.")
    password: str = Field(..., min_length=8, description="Minimum 8 characters.")
    display_name: str | None = Field(default=None, max_length=120, description="Optional display name shown in the UI.")


class LoginRequest(BaseModel):
    email: str = Field(..., description="Registered email address.")
    password: str = Field(..., min_length=1, description="Account password.")


@router.post("/register", response_model=AuthSessionResponse, summary="Register a new account")
def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthSessionResponse:
    """
    Create a new user account and return an access token.

    Returns a bearer token that must be included in the `Authorization` header
    (`Authorization: Bearer <token>`) for all protected endpoints.

    Raises `400` if the email is already in use.
    """
    try:
        session = auth_service.register_user(
            email=str(request.email),
            password=request.password,
            display_name=request.display_name,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_session(session)


@router.post("/login", response_model=AuthSessionResponse, summary="Login and get an access token")
def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthSessionResponse:
    """
    Authenticate with email and password. Returns a bearer access token.

    Raises `401` on invalid credentials.
    """
    try:
        session = auth_service.login_user(
            email=str(request.email),
            password=request.password,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return _serialize_session(session)


@router.get("/me", response_model=AuthUserResponse, summary="Get current authenticated user")
def me(user: UserAccount = Depends(get_current_user)) -> AuthUserResponse:
    """Return the profile of the currently logged-in user. Requires a valid bearer token."""
    return _serialize_user(user)


@router.post("/logout", summary="Logout and invalidate the current token")
def logout(
    user: UserAccount = Depends(get_current_user),
    token: str | None = Depends(get_optional_access_token),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, bool]:
    """
    Invalidate the bearer token so it can no longer be used.

    Returns `{"ok": true}` on success.
    """
    del user
    if token is None:
        raise HTTPException(status_code=400, detail="Missing access token.")
    auth_service.logout(token)
    return {"ok": True}


def _serialize_session(session: AuthenticatedSession) -> AuthSessionResponse:
    return AuthSessionResponse(
        access_token=session.token,
        user=_serialize_user(session.user),
    )


def _serialize_user(user: UserAccount) -> AuthUserResponse:
    return AuthUserResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        tenant_id=str(user.tenant_id),
    )
