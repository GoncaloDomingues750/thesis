from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserInDB(BaseModel):
    id: Optional[str]
    username: str
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
