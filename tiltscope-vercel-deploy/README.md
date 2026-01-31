# 🎯 TILTSCOPE

> **See the tilt before the scoreboard does**

AI-powered esports analytics platform that predicts match outcomes by detecting player "tilt" in real-time.

![TiltScope Dashboard](https://img.shields.io/badge/Status-Live-00ff66?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Cloud9_x_JetBrains-Hackathon_2025-ff0055?style=for-the-badge)

## 🚀 Live Demo

**[View Live Dashboard →](https://tiltscope.vercel.app)**

## 📸 Features

- **Real-time Tilt Detection**: Z-score deviation analysis identifies underperforming players
- **ML Win Predictions**: Ensemble model (LogReg + RF + GBM) predicts match outcomes
- **Performance States**: 🔥 ON FIRE | 📈 HOT | ➖ NORMAL | 📉 COLD | 💀 TILTED
- **What-If Scenarios**: Monte Carlo simulations for alternative outcome analysis
- **Dynamic Dashboard**: Auto-refresh, loading animations, and responsive design

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Backend | Python 3.12, FastAPI, scikit-learn, XGBoost |
| Frontend | React 18, Vanilla CSS, Vercel |
| Data | GRID Esports API, GraphQL |

## 📊 How It Works

1. **Data Collection** → Real-time stats from GRID API
2. **Baseline Calculation** → Build historical K/D mean & std for each player
3. **Z-Score Analysis** → `z = (current_kd - μ) / σ`
4. **Tilt Detection** → Flag players with z < -1.5 as TILTED
5. **ML Prediction** → Ensemble model predicts win probability
6. **What-If Simulation** → "What if OXY wasn't tilted?"

## 🎮 Case Study: Cloud9 vs NRG

**Game 2 Analysis** (Corrode 2-13 NRG):
- Cloud9: OXY (z=-1.70 💀), v1c (z=-1.61 💀)
- NRG: mada (z=+2.79 🔥), brawk (z=+1.24 📈)
- **Prediction**: NRG 96.8% ✓ Correct!

## 🚀 Deploy to Vercel

### Option 1: One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/tiltscope)

### Option 2: Manual Deploy

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to project folder
cd tiltscope-web

# Deploy
vercel

# Or deploy to production
vercel --prod
```

### Option 3: GitHub Integration

1. Push this folder to a GitHub repository
2. Go to [vercel.com](https://vercel.com)
3. Click "Import Project"
4. Select your GitHub repo
5. Click "Deploy"

## 📁 Project Structure

```
tiltscope-web/
├── index.html      # Main dashboard
├── docs.html       # Documentation page
├── vercel.json     # Vercel configuration
└── README.md       # This file
```

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 100% |
| Games Predicted | 14/14 |
| Players Tracked | 25 |

> ⚠️ Note: 100% accuracy on small sample likely indicates overfitting

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/initialize` | Initialize with team data |
| GET | `/api/matches` | List all matches |
| GET | `/api/match/{id}/analysis` | Full match analysis |
| GET | `/api/whatif/{id}/{game}` | What-if scenarios |

## 🏆 Built For

**Cloud9 x JetBrains Hackathon 2025**

---

<p align="center">
  <strong>🎯 TILTSCOPE</strong><br>
  <em>See the tilt before the scoreboard does</em>
</p>
