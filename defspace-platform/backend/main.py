"""Def Space Summer Internship 2026 — Full-Stack Aerodynamics Platform
   FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from database import init_db
from routers import p1, p2, p3, p4, p5, p6

app = FastAPI(
    title="Def Space Internship 2026 — Aerodynamics Platform",
    description=(
        "Full-stack aerodynamics + drone design + GenAI platform. "
        "Student: Nitish Kumar Kannaujiya (24BECCS33), CUJ."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(p1.router)
app.include_router(p2.router)
app.include_router(p3.router)
app.include_router(p4.router)
app.include_router(p5.router)
app.include_router(p6.router)


@app.on_event("startup")
async def on_startup():
    init_db()
    print("✅  Database initialised — Def Space Platform ready")


@app.get("/api/health")
async def health():
    return {"status": "ok", "platform": "Def Space Summer Internship 2026", "version": "1.0.0"}


@app.get("/api/projects")
async def list_projects():
    return [
        {"id": "p1", "title": "Flat Plate Wing CG & Static Stability Analysis",
         "description": "CG calculation, static margin, ML stability scoring"},
        {"id": "p2", "title": "Wing-Tail Neutral Point Analysis",
         "description": "NP computation, tail contribution, stability charts"},
        {"id": "p3", "title": "RC Trainer Aircraft DBT",
         "description": "Design-Build-Test: specs, build log, flight test"},
        {"id": "p4", "title": "Cropped Delta Wing UAV Aerodynamic Analysis",
         "description": "Full alpha sweep, drag polar, MLP neural network, trim analysis"},
        {"id": "p5", "title": "VTOL Drone Design Portal",
         "description": "Propulsion sizing, GBR flight-time prediction, scipy optimiser"},
        {"id": "p6", "title": "GenAI NotebookLM Skill Gap Analyzer",
         "description": "TF-IDF JD-resume matching, skill extraction, 30-day learning plan"},
    ]


# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{page}.html")
    async def serve_page(page: str):
        path = os.path.join(FRONTEND_DIR, f"{page}.html")
        if os.path.exists(path):
            return FileResponse(path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
