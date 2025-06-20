from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from ..models import UserCreate, UserInDB
import os
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi import Depends, HTTPException

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["protein_db"]
users_collection = db["users"]

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def create_user(user: UserCreate) -> UserInDB:
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    result = await users_collection.insert_one(user_dict)
    return UserInDB(id=str(result.inserted_id), **user_dict)


async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    


async def get_user_by_username(username: str) -> UserInDB:
    user = await users_collection.find_one({"username": username})
    if user:
        return UserInDB(id=str(user["_id"]), **user)
    return None