"""
TILTSCOPE - FastAPI Backend
============================
REST API for the TiltScope esports prediction system

Endpoints:
- GET /api/health - Health check
- GET /api/matches - Get recent matches
- GET /api/match/{id}/analysis - Get full analysis for a match
- GET /api/players - Get player baselines
- POST /api/predict - Get prediction for a game
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
import sys

sys.path.insert(0, '.')

from backend.api.grid_client import GRIDClient, GameType
from backend.core.baseline import BaselineCalculator
from backend.core.deviation import DeviationEngine, PerformanceState
from backend.core.predictor import EnsemblePredictor
from backend.core.whatif import WhatIfEngine

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="TiltScope API",
    description="AI-Powered Esports Win Predictor - See the tilt before the scoreboard does",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE (in production, use proper caching/DB)
# ============================================================================

class AppState:
    def __init__(self):
        self.client: Optional[GRIDClient] = None
        self.calculator: Optional[BaselineCalculator] = None
        self.predictor: Optional[EnsemblePredictor] = None
        self.engine: Optional[DeviationEngine] = None
        self.whatif: Optional[WhatIfEngine] = None
        self.matches: List = []
        self.initialized = False

state = AppState()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PlayerStats(BaseModel):
    player_name: str
    team_name: str
    agent: str
    kills: int
    deaths: int
    assists: int
    kd_ratio: float
    z_score: float
    state: str
    baseline_kd: float

class GameAnalysis(BaseModel):
    game_number: int
    map_name: str
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    winner: str
    team1_players: List[PlayerStats]
    team2_players: List[PlayerStats]
    team1_z_score: float
    team2_z_score: float
    prediction: Dict[str, Any]

class MatchAnalysis(BaseModel):
    series_id: str
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    winner: str
    tournament: str
    date: str
    games: List[GameAnalysis]

class PlayerBaseline(BaseModel):
    player_name: str
    team_name: str
    games_played: int
    kd_mean: float
    kd_std: float
    kd_min: float
    kd_max: float
    most_played_agent: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": state.initialized,
        "matches_loaded": len(state.matches)
    }


@app.post("/api/initialize")
async def initialize_system(teams: List[str] = ["Cloud9", "NRG", "Sentinels"]):
    """Initialize the system by fetching match data"""
    try:
        state.client = GRIDClient()
        
        # Fetch matches
        all_matches = []
        for team in teams:
            matches = await state.client.get_team_matches(team, GameType.VALORANT, 3)
            all_matches.extend(matches)
            await asyncio.sleep(0.2)
        
        # Deduplicate
        seen_ids = set()
        state.matches = []
        for m in all_matches:
            if m.series_id not in seen_ids:
                seen_ids.add(m.series_id)
                state.matches.append(m)
        
        # Build baselines and predictor
        state.calculator = BaselineCalculator()
        state.calculator.add_matches(state.matches)
        
        state.engine = DeviationEngine(state.calculator)
        state.predictor = EnsemblePredictor(state.calculator)
        state.predictor.train(state.matches)
        
        state.whatif = WhatIfEngine(state.predictor)
        state.initialized = True
        
        return {
            "status": "initialized",
            "matches_loaded": len(state.matches),
            "players_tracked": len(state.calculator.get_all_baselines())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matches")
async def get_matches():
    """Get list of available matches"""
    if not state.initialized:
        raise HTTPException(status_code=400, detail="System not initialized. Call POST /api/initialize first.")
    
    matches = []
    for m in state.matches:
        matches.append({
            "series_id": m.series_id,
            "team1_name": m.team1_name,
            "team2_name": m.team2_name,
            "team1_score": m.team1_score,
            "team2_score": m.team2_score,
            "winner": m.winner,
            "tournament": m.tournament_name,
            "date": m.date.isoformat(),
            "num_games": len(m.games)
        })
    
    return {"matches": matches}


@app.get("/api/match/{series_id}/analysis")
async def get_match_analysis(series_id: str):
    """Get full analysis for a specific match"""
    if not state.initialized:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Find match
    match = None
    for m in state.matches:
        if m.series_id == series_id:
            match = m
            break
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Analyze each game
    games_analysis = []
    team1_score = 0
    team2_score = 0
    previous_games = []
    
    for game in match.games:
        # Get deviation analysis
        team1_dev, team2_dev = state.engine.analyze_game(game)
        
        # Get prediction
        prediction = state.predictor.predict_game(
            game, team1_score, team2_score, previous_games
        )
        
        # Format player stats
        def format_players(dev):
            players = []
            for p in sorted(dev.player_deviations, key=lambda x: x.z_score, reverse=True):
                players.append(PlayerStats(
                    player_name=p.player_name,
                    team_name=p.team_name,
                    agent=p.agent,
                    kills=p.current_kills,
                    deaths=p.current_deaths,
                    assists=0,  # Would need to add to deviation
                    kd_ratio=round(p.current_kd, 2),
                    z_score=round(p.z_score, 2),
                    state=p.state.value,
                    baseline_kd=round(p.baseline_kd_mean, 2)
                ))
            return players
        
        games_analysis.append(GameAnalysis(
            game_number=game.game_number,
            map_name=game.map_name,
            team1_name=game.team1_name,
            team2_name=game.team2_name,
            team1_score=game.team1_score,
            team2_score=game.team2_score,
            winner=game.winner,
            team1_players=format_players(team1_dev),
            team2_players=format_players(team2_dev),
            team1_z_score=round(team1_dev.average_z_score, 2),
            team2_z_score=round(team2_dev.average_z_score, 2),
            prediction={
                "team1_prob": round(prediction.team1_win_prob * 100, 1),
                "team2_prob": round(prediction.team2_win_prob * 100, 1),
                "confidence": round(prediction.confidence * 100, 1),
                "predicted_winner": prediction.predicted_winner,
                "actual_winner": game.winner,
                "correct": prediction.predicted_winner == game.winner,
                "top_factors": [f[0] for f in prediction.top_factors[:3]]
            }
        ))
        
        # Update for next game
        if game.winner == match.team1_name:
            team1_score += 1
        else:
            team2_score += 1
        previous_games.append(game)
    
    return MatchAnalysis(
        series_id=match.series_id,
        team1_name=match.team1_name,
        team2_name=match.team2_name,
        team1_score=match.team1_score,
        team2_score=match.team2_score,
        winner=match.winner,
        tournament=match.tournament_name,
        date=match.date.isoformat(),
        games=games_analysis
    )


@app.get("/api/players")
async def get_players():
    """Get all player baselines"""
    if not state.initialized:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    baselines = state.calculator.get_all_baselines()
    
    players = []
    for b in sorted(baselines, key=lambda x: x.kd_mean, reverse=True):
        players.append(PlayerBaseline(
            player_name=b.player_name,
            team_name=b.team_name,
            games_played=b.games_played,
            kd_mean=round(b.kd_mean, 2),
            kd_std=round(b.kd_std_dev, 2),
            kd_min=round(b.kd_min, 2),
            kd_max=round(b.kd_max, 2),
            most_played_agent=b.most_played_agent
        ))
    
    return {"players": players}


@app.get("/api/whatif/{series_id}/{game_number}")
async def get_whatif_analysis(series_id: str, game_number: int):
    """Get what-if scenarios for a specific game"""
    if not state.initialized:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Find match and game
    match = None
    for m in state.matches:
        if m.series_id == series_id:
            match = m
            break
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    game = None
    for g in match.games:
        if g.game_number == game_number:
            game = g
            break
    
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get previous games for context
    previous_games = [g for g in match.games if g.game_number < game_number]
    team1_score = sum(1 for g in previous_games if g.winner == match.team1_name)
    team2_score = len(previous_games) - team1_score
    
    # Run scenarios
    results = state.whatif.run_all_scenarios(game, team1_score, team2_score, previous_games)
    
    scenarios = []
    for r in results[:5]:
        scenarios.append({
            "description": r.scenario.description,
            "original_prob": round(r.original_team1_prob * 100, 1),
            "simulated_prob": round(r.simulated_team1_prob * 100, 1),
            "probability_shift": round(r.probability_shift * 100, 1),
            "outcome_changed": r.outcome_changed,
            "impact": r.impact_description
        })
    
    return {
        "game_number": game_number,
        "map_name": game.map_name,
        "scenarios": scenarios
    }


# ============================================================================
# DEMO DATA ENDPOINT (for frontend development)
# ============================================================================

@app.get("/api/demo")
async def get_demo_data():
    """Returns demo data for frontend development without needing GRID API"""
    return {
        "match": {
            "series_id": "2843069",
            "team1_name": "Cloud9",
            "team2_name": "NRG",
            "team1_score": 1,
            "team2_score": 2,
            "winner": "NRG",
            "tournament": "VCT Americas - Stage 2 2025",
            "games": [
                {
                    "game_number": 1,
                    "map_name": "haven",
                    "team1_score": 13,
                    "team2_score": 10,
                    "winner": "Cloud9",
                    "team1_z_score": 0.62,
                    "team2_z_score": -0.30,
                    "prediction": {"team1_prob": 97.0, "team2_prob": 3.0, "correct": True},
                    "team1_players": [
                        {"player_name": "OXY", "agent": "neon", "kills": 24, "deaths": 16, "kd_ratio": 1.50, "z_score": 0.70, "state": "📈 Hot", "baseline_kd": 1.19},
                        {"player_name": "v1c", "agent": "omen", "kills": 22, "deaths": 15, "kd_ratio": 1.47, "z_score": 1.08, "state": "📈 Hot", "baseline_kd": 1.08},
                        {"player_name": "Xeppaa", "agent": "vyse", "kills": 21, "deaths": 18, "kd_ratio": 1.17, "z_score": 1.26, "state": "📈 Hot", "baseline_kd": 0.83},
                        {"player_name": "neT", "agent": "viper", "kills": 18, "deaths": 15, "kd_ratio": 1.20, "z_score": 1.27, "state": "📈 Hot", "baseline_kd": 0.71},
                        {"player_name": "mitch", "agent": "skye", "kills": 4, "deaths": 16, "kd_ratio": 0.25, "z_score": -1.23, "state": "📉 Cold", "baseline_kd": 0.65}
                    ],
                    "team2_players": [
                        {"player_name": "s0m", "agent": "viper", "kills": 19, "deaths": 16, "kd_ratio": 1.19, "z_score": 1.02, "state": "📈 Hot", "baseline_kd": 0.90},
                        {"player_name": "skuba", "agent": "killjoy", "kills": 17, "deaths": 16, "kd_ratio": 1.06, "z_score": -0.31, "state": "➖ Normal", "baseline_kd": 1.16},
                        {"player_name": "mada", "agent": "neon", "kills": 15, "deaths": 20, "kd_ratio": 0.75, "z_score": -0.46, "state": "➖ Normal", "baseline_kd": 1.24},
                        {"player_name": "Ethan", "agent": "omen", "kills": 16, "deaths": 20, "kd_ratio": 0.80, "z_score": -0.80, "state": "📉 Cold", "baseline_kd": 1.16},
                        {"player_name": "brawk", "agent": "sova", "kills": 13, "deaths": 17, "kd_ratio": 0.76, "z_score": -0.98, "state": "📉 Cold", "baseline_kd": 1.56}
                    ]
                },
                {
                    "game_number": 2,
                    "map_name": "corrode",
                    "team1_score": 2,
                    "team2_score": 13,
                    "winner": "NRG",
                    "team1_z_score": -0.97,
                    "team2_z_score": 1.02,
                    "prediction": {"team1_prob": 3.2, "team2_prob": 96.8, "correct": True},
                    "team1_players": [
                        {"player_name": "mitch", "agent": "skye", "kills": 7, "deaths": 14, "kd_ratio": 0.50, "z_score": -0.47, "state": "➖ Normal", "baseline_kd": 0.65},
                        {"player_name": "Xeppaa", "agent": "vyse", "kills": 9, "deaths": 13, "kd_ratio": 0.69, "z_score": -0.52, "state": "📉 Cold", "baseline_kd": 0.83},
                        {"player_name": "neT", "agent": "viper", "kills": 6, "deaths": 12, "kd_ratio": 0.50, "z_score": -0.56, "state": "📉 Cold", "baseline_kd": 0.71},
                        {"player_name": "v1c", "agent": "omen", "kills": 7, "deaths": 14, "kd_ratio": 0.50, "z_score": -1.61, "state": "💀 TILTED", "baseline_kd": 1.08},
                        {"player_name": "OXY", "agent": "neon", "kills": 6, "deaths": 14, "kd_ratio": 0.43, "z_score": -1.70, "state": "💀 TILTED", "baseline_kd": 1.19}
                    ],
                    "team2_players": [
                        {"player_name": "mada", "agent": "waylay", "kills": 21, "deaths": 5, "kd_ratio": 4.20, "z_score": 2.79, "state": "🔥 ON FIRE", "baseline_kd": 1.24},
                        {"player_name": "brawk", "agent": "sova", "kills": 18, "deaths": 7, "kd_ratio": 2.57, "z_score": 1.24, "state": "📈 Hot", "baseline_kd": 1.56},
                        {"player_name": "s0m", "agent": "omen", "kills": 7, "deaths": 6, "kd_ratio": 1.17, "z_score": 0.94, "state": "📈 Hot", "baseline_kd": 0.90},
                        {"player_name": "Ethan", "agent": "kay/o", "kills": 13, "deaths": 9, "kd_ratio": 1.44, "z_score": 0.62, "state": "📈 Hot", "baseline_kd": 1.16},
                        {"player_name": "skuba", "agent": "viper", "kills": 8, "deaths": 8, "kd_ratio": 1.00, "z_score": -0.50, "state": "➖ Normal", "baseline_kd": 1.16}
                    ]
                },
                {
                    "game_number": 3,
                    "map_name": "lotus",
                    "team1_score": 10,
                    "team2_score": 13,
                    "winner": "NRG",
                    "team1_z_score": -0.24,
                    "team2_z_score": 0.16,
                    "prediction": {"team1_prob": 5.2, "team2_prob": 94.8, "correct": True},
                    "team1_players": [
                        {"player_name": "v1c", "agent": "omen", "kills": 20, "deaths": 16, "kd_ratio": 1.25, "z_score": 0.48, "state": "➖ Normal", "baseline_kd": 1.08},
                        {"player_name": "OXY", "agent": "raze", "kills": 22, "deaths": 19, "kd_ratio": 1.16, "z_score": -0.07, "state": "➖ Normal", "baseline_kd": 1.19},
                        {"player_name": "mitch", "agent": "fade", "kills": 11, "deaths": 18, "kd_ratio": 0.61, "z_score": -0.13, "state": "➖ Normal", "baseline_kd": 0.65},
                        {"player_name": "Xeppaa", "agent": "vyse", "kills": 13, "deaths": 19, "kd_ratio": 0.68, "z_score": -0.55, "state": "📉 Cold", "baseline_kd": 0.83},
                        {"player_name": "neT", "agent": "viper", "kills": 7, "deaths": 20, "kd_ratio": 0.35, "z_score": -0.95, "state": "📉 Cold", "baseline_kd": 0.71}
                    ],
                    "team2_players": [
                        {"player_name": "s0m", "agent": "omen", "kills": 17, "deaths": 13, "kd_ratio": 1.31, "z_score": 1.44, "state": "📈 Hot", "baseline_kd": 0.90},
                        {"player_name": "Ethan", "agent": "fade", "kills": 23, "deaths": 16, "kd_ratio": 1.44, "z_score": 0.60, "state": "📈 Hot", "baseline_kd": 1.16},
                        {"player_name": "brawk", "agent": "vyse", "kills": 21, "deaths": 12, "kd_ratio": 1.75, "z_score": 0.23, "state": "➖ Normal", "baseline_kd": 1.56},
                        {"player_name": "mada", "agent": "raze", "kills": 19, "deaths": 16, "kd_ratio": 1.19, "z_score": -0.05, "state": "➖ Normal", "baseline_kd": 1.24},
                        {"player_name": "skuba", "agent": "viper", "kills": 11, "deaths": 16, "kd_ratio": 0.69, "z_score": -1.46, "state": "📉 Cold", "baseline_kd": 1.16}
                    ]
                }
            ]
        },
        "player_progression": [
            {"name": "v1c", "team": "Cloud9", "trajectory": [{"game": 1, "kd": 1.47, "state": "hot"}, {"game": 2, "kd": 0.50, "state": "tilted"}, {"game": 3, "kd": 1.25, "state": "normal"}]},
            {"name": "OXY", "team": "Cloud9", "trajectory": [{"game": 1, "kd": 1.50, "state": "hot"}, {"game": 2, "kd": 0.43, "state": "tilted"}, {"game": 3, "kd": 1.16, "state": "normal"}]},
            {"name": "mitch", "team": "Cloud9", "trajectory": [{"game": 1, "kd": 0.25, "state": "cold"}, {"game": 2, "kd": 0.50, "state": "normal"}, {"game": 3, "kd": 0.61, "state": "normal"}]},
            {"name": "neT", "team": "Cloud9", "trajectory": [{"game": 1, "kd": 1.20, "state": "hot"}, {"game": 2, "kd": 0.50, "state": "cold"}, {"game": 3, "kd": 0.35, "state": "cold"}]},
            {"name": "mada", "team": "NRG", "trajectory": [{"game": 1, "kd": 0.75, "state": "normal"}, {"game": 2, "kd": 4.20, "state": "fire"}, {"game": 3, "kd": 1.19, "state": "normal"}]},
            {"name": "brawk", "team": "NRG", "trajectory": [{"game": 1, "kd": 0.76, "state": "cold"}, {"game": 2, "kd": 2.57, "state": "hot"}, {"game": 3, "kd": 1.75, "state": "normal"}]}
        ]
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
