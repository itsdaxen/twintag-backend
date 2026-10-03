from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from twintag_backend.api.scans import router as scans_router


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


def create_app() -> FastAPI:
    app = FastAPI(title="TwinTag API", version="0.1.0")
    app.include_router(scans_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="twintag-backend")

    return app


app = create_app()


def run() -> None:
    uvicorn.run("twintag_backend.main:app", host="127.0.0.1", port=8000, reload=True)
