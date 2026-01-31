"""
Feature Extractor for ML Prediction
Transforms match/game data into numerical features for the ML model

Features include:
- Team-level aggregations (avg KD, total kills, etc.)
- Player deviation features (z-scores, tilt count)
- Historical features (win rates, head-to-head)
- Agent/composition features
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics

from .baseline import BaselineCalculator, PlayerBaseline
from .deviation import DeviationEngine, PerformanceState


@dataclass
class MatchFeatures:
    """Feature vector for a match prediction"""
    
    # Team 1 features
    team1_avg_kd: float = 0.0
    team1_avg_z_score: float = 0.0
    team1_tilted_count: int = 0
    team1_hot_count: int = 0
    team1_total_kills: int = 0
    team1_total_deaths: int = 0
    team1_first_kills: int = 0
    
    # Team 2 features  
    team2_avg_kd: float = 0.0
    team2_avg_z_score: float = 0.0
    team2_tilted_count: int = 0
    team2_hot_count: int = 0
    team2_total_kills: int = 0
    team2_total_deaths: int = 0
    team2_first_kills: int = 0
    
    # Differential features (team1 - team2)
    kd_diff: float = 0.0
    z_score_diff: float = 0.0
    momentum_diff: float = 0.0  # Based on recent performance trend
    
    # Game context
    map_name: str = ""
    game_number: int = 1
    team1_series_score: int = 0
    team2_series_score: int = 0
    
    # Target (for training)
    team1_won: Optional[bool] = None
    
    def to_vector(self) -> List[float]:
        """Convert to numerical feature vector for ML"""
        return [
            self.team1_avg_kd,
            self.team1_avg_z_score,
            self.team1_tilted_count,
            self.team1_hot_count,
            self.team1_total_kills,
            self.team1_total_deaths,
            self.team1_first_kills,
            self.team2_avg_kd,
            self.team2_avg_z_score,
            self.team2_tilted_count,
            self.team2_hot_count,
            self.team2_total_kills,
            self.team2_total_deaths,
            self.team2_first_kills,
            self.kd_diff,
            self.z_score_diff,
            self.momentum_diff,
            self.game_number,
            self.team1_series_score,
            self.team2_series_score,
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretability"""
        return [
            "team1_avg_kd",
            "team1_avg_z_score", 
            "team1_tilted_count",
            "team1_hot_count",
            "team1_total_kills",
            "team1_total_deaths",
            "team1_first_kills",
            "team2_avg_kd",
            "team2_avg_z_score",
            "team2_tilted_count", 
            "team2_hot_count",
            "team2_total_kills",
            "team2_total_deaths",
            "team2_first_kills",
            "kd_diff",
            "z_score_diff",
            "momentum_diff",
            "game_number",
            "team1_series_score",
            "team2_series_score",
        ]


class FeatureExtractor:
    """
    Extracts ML features from game data
    
    Usage:
        extractor = FeatureExtractor(baseline_calculator)
        features = extractor.extract_game_features(game_result)
    """
    
    def __init__(self, calculator: BaselineCalculator):
        self.calculator = calculator
        self.engine = DeviationEngine(calculator)
    
    def extract_game_features(
        self, 
        game_result,
        team1_series_score: int = 0,
        team2_series_score: int = 0,
        previous_games: List = None
    ) -> MatchFeatures:
        """
        Extract features from a single game result
        
        Args:
            game_result: GameResult object
            team1_series_score: Current series score for team 1
            team2_series_score: Current series score for team 2
            previous_games: List of previous GameResult objects for momentum
            
        Returns:
            MatchFeatures object
        """
        # Get deviation analysis
        team1_dev, team2_dev = self.engine.analyze_game(game_result)
        
        # Team 1 stats
        team1_stats = self._aggregate_team_stats(game_result, game_result.team1_name)
        team1_kds = [p.current_kd for p in team1_dev.player_deviations]
        team1_zs = [p.z_score for p in team1_dev.player_deviations]
        
        # Team 2 stats
        team2_stats = self._aggregate_team_stats(game_result, game_result.team2_name)
        team2_kds = [p.current_kd for p in team2_dev.player_deviations]
        team2_zs = [p.z_score for p in team2_dev.player_deviations]
        
        # Calculate momentum from previous games
        momentum_diff = self._calculate_momentum(previous_games, game_result.team1_name, game_result.team2_name)
        
        features = MatchFeatures(
            # Team 1
            team1_avg_kd=statistics.mean(team1_kds) if team1_kds else 1.0,
            team1_avg_z_score=statistics.mean(team1_zs) if team1_zs else 0.0,
            team1_tilted_count=len(team1_dev.tilted_players),
            team1_hot_count=len(team1_dev.hot_players),
            team1_total_kills=team1_stats["kills"],
            team1_total_deaths=team1_stats["deaths"],
            team1_first_kills=team1_stats["first_kills"],
            
            # Team 2
            team2_avg_kd=statistics.mean(team2_kds) if team2_kds else 1.0,
            team2_avg_z_score=statistics.mean(team2_zs) if team2_zs else 0.0,
            team2_tilted_count=len(team2_dev.tilted_players),
            team2_hot_count=len(team2_dev.hot_players),
            team2_total_kills=team2_stats["kills"],
            team2_total_deaths=team2_stats["deaths"],
            team2_first_kills=team2_stats["first_kills"],
            
            # Differentials
            kd_diff=(statistics.mean(team1_kds) if team1_kds else 1.0) - (statistics.mean(team2_kds) if team2_kds else 1.0),
            z_score_diff=(statistics.mean(team1_zs) if team1_zs else 0.0) - (statistics.mean(team2_zs) if team2_zs else 0.0),
            momentum_diff=momentum_diff,
            
            # Context
            map_name=game_result.map_name,
            game_number=game_result.game_number,
            team1_series_score=team1_series_score,
            team2_series_score=team2_series_score,
            
            # Target
            team1_won=(game_result.winner == game_result.team1_name)
        )
        
        return features
    
    def _aggregate_team_stats(self, game_result, team_name: str) -> Dict:
        """Aggregate stats for a team from game result"""
        stats = {"kills": 0, "deaths": 0, "assists": 0, "first_kills": 0}
        
        for p in game_result.player_stats:
            if p.team_name == team_name:
                stats["kills"] += p.kills
                stats["deaths"] += p.deaths
                stats["assists"] += p.assists
                if p.first_kill:
                    stats["first_kills"] += 1
        
        return stats
    
    def _calculate_momentum(self, previous_games: List, team1_name: str, team2_name: str) -> float:
        """
        Calculate momentum differential based on recent game performance
        
        Positive = team1 has momentum, Negative = team2 has momentum
        """
        if not previous_games:
            return 0.0
        
        team1_momentum = 0.0
        team2_momentum = 0.0
        
        # Weight recent games more heavily
        weights = [0.5, 0.3, 0.2]  # Most recent game weighted highest
        
        for i, game in enumerate(reversed(previous_games[-3:])):
            weight = weights[i] if i < len(weights) else 0.1
            
            if game.winner == team1_name:
                team1_momentum += weight
            elif game.winner == team2_name:
                team2_momentum += weight
        
        return team1_momentum - team2_momentum
    
    def extract_match_features(self, match_result) -> List[MatchFeatures]:
        """
        Extract features for all games in a match
        
        Returns list of MatchFeatures, one per game
        """
        features_list = []
        
        team1_score = 0
        team2_score = 0
        previous_games = []
        
        for game in match_result.games:
            features = self.extract_game_features(
                game,
                team1_series_score=team1_score,
                team2_series_score=team2_score,
                previous_games=previous_games
            )
            features_list.append(features)
            
            # Update series score
            if game.winner == match_result.team1_name:
                team1_score += 1
            else:
                team2_score += 1
            
            previous_games.append(game)
        
        return features_list
