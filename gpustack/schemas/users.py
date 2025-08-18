from datetime import datetime
import re
from enum import Enum
from sqlalchemy import Enum as SQLEnum, Text
from typing import Optional
from pydantic import field_validator
from sqlmodel import Column, Field, SQLModel

from .common import PaginatedList
from ..mixins import BaseModelMixin


class AuthProviderEnum(str, Enum):
    Local = "Local"
    OIDC = "OIDC"
    SAML = "SAML"


class UserBase(SQLModel):
    username: str
    is_admin: bool = False
    full_name: Optional[str] = None
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    source: Optional[str] = Field(
        default=AuthProviderEnum.Local, sa_type=SQLEnum(AuthProviderEnum)
    )
    require_password_change: bool = Field(default=False)


class UserCreate(UserBase):
    password: str

    @field_validator('password')
    def validate_password(cls, value):
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', value):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*_+]', value):
            raise ValueError('Password must contain at least one special character')
        return value


class UserUpdate(UserBase):
    password: Optional[str] = None


class UpdatePassword(SQLModel):
    current_password: str
    new_password: str

    @field_validator('new_password')
    def validate_password(cls, value):
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', value):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*_+]', value):
            raise ValueError('Password must contain at least one special character')
        return value


class User(UserBase, BaseModelMixin, table=True):
    __tablename__ = 'users'
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str


class UserPublic(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime


UsersPublic = PaginatedList[UserPublic]
