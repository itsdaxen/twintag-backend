from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from twintag_backend.api.scans import router as scans_router
from twintag_backend.api.synthetic import router as synthetic_router


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


def create_app() -> FastAPI:
    app = FastAPI(title="TwinTag API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(scans_router)
    app.include_router(synthetic_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="twintag-backend")

    return app


app = create_app()


def run() -> None:
    uvicorn.run("twintag_backend.main:app", host="127.0.0.1", port=8000, reload=True)
