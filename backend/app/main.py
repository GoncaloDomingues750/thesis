from fastapi import FastAPI
from app.routes import router as app_router
from fastapi.middleware.cors import CORSMiddleware
from app.auth.auth_routes import router as auth_router


app = FastAPI()

# Allow frontend at localhost:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # <- allow specific origins
    allow_credentials=True,
    allow_methods=["*"],            # <- allow all methods (POST, GET, etc.)
    allow_headers=["*"],            # <- allow all headers
)


app.include_router(auth_router)
app.include_router(app_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Protein Structure Prediction API"}