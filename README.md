<p align="center">
  <img src="https://img.shields.io/badge/🎯-TILTSCOPE-ff0055?style=for-the-badge&labelColor=0a0a12" alt="TiltScope"/>
</p>

<h1 align="center">TILTSCOPE</h1>
<h3 align="center">See the tilt before the scoreboard does</h3>

<p align="center">
  <strong>AI-powered esports analytics platform that predicts match outcomes by detecting player "tilt" in real-time</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Category-Assistant%20Coach-00f0ff?style=flat-square" alt="Category"/>
  <img src="https://img.shields.io/badge/Game-VALORANT-ff4655?style=flat-square" alt="VALORANT"/>
  <img src="https://img.shields.io/badge/Data-GRID%20API-00ff66?style=flat-square" alt="GRID"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-the-problem">Problem</a> •
  <a href="#-solution">Solution</a> •
  <a href="#%EF%B8%8F-how-it-works">How It Works</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-quick-start">Quick Start</a>
</p>

---

## 🏆 Cloud9 x JetBrains Hackathon 2025

**Category:** Category 1 - Comprehensive Assistant Coach  
**Submission:** Sky's the Limit Hackathon  
**Built with:** JetBrains IDE + GRID Esports API

---

## 🚀 Live Demo

**[→ View Live Dashboard](https://tiltscope.vercel.app)**

---

## 🎯 The Problem

Traditional esports analytics tell you **what happened** — but not **why** or **when to act**.

By the time a player's K/D ratio drops, the damage is done. Coaches and analysts need **predictive insights**, not post-game autopsies.

**Key Pain Points:**
- K/D ratios are **lagging indicators** — they show decline after it's too late
- No way to detect **mental state changes** (tilt) during a match
- Coaches lack tools to answer **"what if"** questions about strategy

---

## 💡 Solution

TiltScope detects **player tilt in real-time** using statistical deviation analysis, then predicts its impact on match outcomes with ML.

### Core Innovation: Z-Score Tilt Detection

Instead of looking at raw K/D, we compare **current performance to each player's historical baseline**:

```
z_score = (current_kd - player_mean) / player_std_dev
```

This reveals:
- 🔥 **ON FIRE** (z > +1.5): Player performing 50%+ above their baseline
- 📈 **HOT** (+0.5 < z < +1.5): Above average game
- ➖ **NORMAL** (-0.5 < z < +0.5): Playing at baseline
- 📉 **COLD** (-1.5 < z < -0.5): Below average, watch closely
- 💀 **TILTED** (z < -1.5): Player is tilting — intervene NOW

---

## ⚡️ How It Works

### Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  GRID API   │───▶│   Baseline   │───▶│  Z-Score      │───▶│  ML Ensemble │
│  Match Data │    │   Calculator │    │  Detection    │    │  Predictor   │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                              ┌──────────────────────────────┐
                                              │  What-If Monte Carlo Engine  │
                                              │  "What if OXY wasn't tilted?" │
                                              └──────────────────────────────┘
```

### Step-by-Step Process

| Step | Component | Description |
|------|-----------|-------------|
| 1️⃣ | **Data Collection** | Fetch real-time match data from GRID Esports API (kills, deaths, assists per round) |
| 2️⃣ | **Baseline Calculation** | Build historical K/D mean (μ) and standard deviation (σ) for each player |
| 3️⃣ | **Z-Score Analysis** | Calculate deviation: `z = (current - μ) / σ` |
| 4️⃣ | **State Classification** | Classify players: FIRE / HOT / NORMAL / COLD / TILTED |
| 5️⃣ | **ML Prediction** | Ensemble model predicts win probability based on team states |
| 6️⃣ | **What-If Simulation** | Monte Carlo engine answers strategic questions |

---

## 🤖 ML Model Architecture

### Ensemble Prediction (Weighted Voting)

```python
prediction = (
    0.25 * LogisticRegression +
    0.35 * RandomForest +
    0.40 * GradientBoosting
)
```

### Feature Engineering (20 features per game)

| Feature Type | Examples |
|--------------|----------|
| **Team Averages** | avg_kd, avg_z_score, total_kills, total_deaths |
| **State Counts** | tilted_count, hot_count, fire_count |
| **Differentials** | kd_diff, z_score_diff, momentum |
| **Context** | map_encoding, game_number, series_score |

### Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 100% |
| Games Predicted Correctly | 14/14 |
| Players Tracked | 25 |

> ⚠️ **Note:** 100% accuracy on small sample likely indicates overfitting. Real-world deployment would require larger validation set.

---

## 🎮 Case Study: Cloud9 vs NRG

### Game 2 (Corrode) — The Tilt Game

| Team | Z-Score | Outcome |
|------|---------|---------|
| Cloud9 | **-0.97** (Team Tilted) | Lost 2-13 |
| NRG | **+1.02** (Team Hot) | Won 13-2 |

**Key Tilt Detections:**

| Player | K/D | Z-Score | State | Impact |
|--------|-----|---------|-------|--------|
| OXY (C9) | 0.43 | -1.70 | 💀 TILTED | Star player collapsed |
| v1c (C9) | 0.50 | -1.61 | 💀 TILTED | Secondary carry tilted |
| mada (NRG) | 4.20 | +2.79 | 🔥 ON FIRE | Explosive carry performance |

**TiltScope Prediction:** NRG 96.8% → ✅ **CORRECT**

### What-If Analysis

> "What if OXY had performed at baseline instead of tilting?"

Monte Carlo Simulation (100 iterations):
- Original NRG win probability: 96.8%
- Simulated with OXY at baseline: **62.3%**
- **Tilt Impact:** OXY's tilt cost Cloud9 ~35% win probability

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.12, FastAPI, asyncio, httpx |
| **ML/Data** | scikit-learn, XGBoost, pandas, NumPy |
| **Frontend** | React 18, Vanilla CSS, Vercel |
| **Data Source** | GRID Esports API (VALORANT) |
| **IDE** | JetBrains PyCharm |

---

## 📁 Project Structure

```
tiltscope/
├── backend/
│   ├── main.py              # FastAPI server & REST endpoints
│   ├── api/
│   │   ├── __init__.py
│   │   └── grid_client.py   # GRID API integration
│   └── core/
│       ├── __init__.py
│       ├── baseline.py      # Player baseline calculator
│       ├── deviation.py     # Tilt detection engine
│       ├── features.py      # ML feature engineering
│       ├── predictor.py     # Ensemble ML predictor
│       └── whatif.py        # Monte Carlo simulator
├── frontend/
│   ├── index.html           # Main dashboard
│   └── docs.html            # Documentation page
├── requirements.txt         # Python dependencies
├── vercel.json             # Vercel deployment config
├── LICENSE                  # MIT License
└── README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GRID API Key ([Apply here](https://grid.gg/hackathon-application-form/))

### Backend Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tiltscope.git
cd tiltscope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GRID_API_KEY="your_api_key_here"

# Run server
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
# Option 1: Open directly
open frontend/index.html

# Option 2: Serve locally
cd frontend
python -m http.server 3000
# Visit http://localhost:3000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/initialize` | Initialize with team data |
| GET | `/api/matches` | List available matches |
| GET | `/api/match/{id}/analysis` | Full match analysis |
| GET | `/api/whatif/{id}/{game}` | What-if scenarios |
| GET | `/api/demo` | Demo data (no API key needed) |

---

## 🌐 Deploy to Vercel

### One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/tiltscope)

### Manual Deploy

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

---

## 📊 Dashboard Features

- **🔄 Refresh Data** — Simulate real-time data reload with animations
- **⚡ Auto-Refresh** — Toggle 30-second automatic updates
- **🎮 Game Tabs** — Switch between games in a series
- **📈 Player Cards** — Animated cards showing K/D, z-score, state
- **🎯 ML Predictions** — Win probability with confidence visualization
- **📊 Tilt Progression** — Track player performance across games
- **🌙 Cyberpunk Theme** — Dark mode with neon accents

---

## 🎥 Demo Video

[**→ Watch 3-Minute Demo**](https://youtube.com/your-video-link)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Cloud9** — For hosting an amazing hackathon
- **JetBrains** — For incredible developer tools
- **GRID** — For providing official esports data
- **Moneyball** — For the inspiration (Peter Brand would be proud)

---

<p align="center">
  <strong>🎯 TILTSCOPE</strong><br>
  <em>See the tilt before the scoreboard does</em>
</p>

<p align="center">
  Built with ❤️ for the Cloud9 x JetBrains Hackathon 2025
</p>
