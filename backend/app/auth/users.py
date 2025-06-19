from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from ..models import UserCreate, UserInDB
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["protein_db"]
users_collection = db["users"]

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

async def get_user_by_username(username: str) -> UserInDB:
    user = await users_collection.find_one({"username": username})
    if user:
        return UserInDB(id=str(user["_id"]), **user)
    return None