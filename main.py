import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from ai.router import router as ai_router
from ws.ws_router import router as ws_router
from sk.fingertip_router import router as fingertip_router


def create_app():
    app = FastAPI(
        title="Dot-Stem AI Image Analysis API",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8080",
            "http://localhost:8081",
            "https://dotstem-ui.web.app",
            "http://34.97.145.226",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # FastAPI 웹소켓 라우터는 이미 app.include_router()로 등록 가능함
    app.include_router(ai_router, tags=["Index"])
    app.include_router(ws_router, tags=["WebSocket"])
    app.include_router(fingertip_router, tags=["FingerTip"])

    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
