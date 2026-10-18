# Def Space Aerodynamics Platform — Summer Internship 2026

**Student:** Nitish Kumar Kannaujiya · 24BECCS33  
**Course:** B.Tech CSE (Cybersecurity) · 5th Semester  
**University:** Central University of Jammu  
**Internship:** Def Space Summer Internship 2026

---

## Overview

Full-stack aerodynamics & drone design platform covering 6 internship projects.  
Single unified FastAPI backend · SQLite database · Vanilla JS frontend · Docker deployment.

---

## Projects

| # | Title | Key Tech |
|---|-------|----------|
| P1 | CG & Static Stability Analysis | LinearRegression, NumPy moments |
| P2 | Wing-Tail Neutral Point Analysis | PolynomialFeatures + LinearRegression |
| P3 | RC Trainer Aircraft DBT | Build/Flight log CRUD, performance physics |
| P4 | Cropped Delta Wing UAV Aerodynamics | MLP (64→32), full alpha sweep, drag polar |
| P5 | VTOL Drone Design Portal | GradientBoosting, SciPy L-BFGS-B optimiser |
| P6 | GenAI Skill Gap Analyzer | TF-IDF cosine similarity, 30-day plan |

---

## Tech Stack

**Backend**
- Python 3.11 · FastAPI 0.115 · Uvicorn
- SQLAlchemy 2.0 ORM · SQLite
- NumPy 2.1 · SciPy 1.14 · scikit-learn 1.5
- Pydantic v2 · REST JSON API · OpenAPI/Swagger

**Frontend**
- Vanilla JS · HTML5 · CSS3 (custom design system)
- Chart.js 4.4.1
- Inter + JetBrains Mono fonts
- Dark engineering/avionics aesthetic

**Deployment**
- Docker + docker-compose
- Nginx 1.25 reverse proxy

---

## Quick Start (Local Development)

```bash
# Clone the repo
git clone https://github.com/nitishkumar9898/defspace-internship-platform.git
cd defspace-internship-platform

# Install Python dependencies
pip install -r backend/requirements.txt

# Start the backend (serves frontend too)
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

**API docs (Swagger):** http://localhost:8000/api/docs

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Platform available at http://localhost:80

---

## Project Structure

```
defspace-platform/
├── backend/
│   ├── main.py              # FastAPI app + static file serving
│   ├── database.py          # SQLAlchemy engine + session
│   ├── models.py            # 6 ORM table models
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── requirements.txt
│   └── routers/
│       ├── p1.py            # CG & stability (LinearRegression)
│       ├── p2.py            # Neutral point (PolyReg)
│       ├── p3.py            # RC trainer specs + log CRUD
│       ├── p4.py            # Delta wing aero (MLP)
│       ├── p5.py            # VTOL drone (GBR + SciPy optimiser)
│       └── p6.py            # Skill analyzer (TF-IDF)
├── frontend/
│   ├── index.html           # Main dashboard
│   ├── p1.html – p6.html   # Project pages
│   ├── css/
│   │   └── design.css       # Complete design system
│   └── js/
│       └── api.js           # API client + helpers
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/projects` | List all projects |
| POST | `/api/p1/calculate` | CG + static margin calculation |
| GET | `/api/p1/history` | Last 50 P1 results |
| GET | `/api/p1/sensitivity` | NP sensitivity chart data |
| POST | `/api/p2/calculate` | Neutral point + stability charts |
| GET | `/api/p2/model-info` | ML model metadata |
| POST | `/api/p3/analyse` | RC aircraft performance |
| POST | `/api/p3/log` | Add build/flight log entry |
| GET | `/api/p3/logs` | Get log entries (filterable) |
| GET | `/api/p3/specs` | Fixed CUJ trainer specs |
| POST | `/api/p4/calculate` | Full alpha sweep + MLP predictions |
| GET | `/api/p4/model-info` | MLP metadata |
| POST | `/api/p5/calculate` | VTOL propulsion + optimiser |
| GET | `/api/p5/model-info` | GBR metadata |
| POST | `/api/p6/analyse` | Skill gap analysis |
| GET | `/api/p6/history` | Analysis history |

Full interactive docs: `/api/docs` (Swagger UI)

---

## ML Models

| Project | Algorithm | Training Data | R² Target |
|---------|-----------|--------------|-----------|
| P1 | LinearRegression | 200 synthetic stability samples | ~95% |
| P2 | PolynomialFeatures (deg=2) + LR | 300 aero samples | ~97% |
| P4 | MLPRegressor (64→32) | 2000 NACA synthetic samples | ~93% |
| P5 | GradientBoostingRegressor (200 est) | 3000 drone records | ~94% |
| P6 | TF-IDF + cosine similarity | — (inference only) | — |

---

## License

Academic / educational use — Def Space Summer Internship 2026.  
Central University of Jammu · Department of Computer Science & Engineering.
