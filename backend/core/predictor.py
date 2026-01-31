"""
TiltScope ML Ensemble Predictor
================================
Combines multiple ML models for robust win probability prediction:
- Logistic Regression (interpretable baseline)
- Random Forest (handles non-linear relationships)
- Gradient Boosting (captures complex patterns)

The ensemble weights predictions based on model confidence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum

# Try to import sklearn, but provide fallback
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - using simplified predictor")

from .features import FeatureExtractor, MatchFeatures
from .baseline import BaselineCalculator


@dataclass
class PredictionResult:
    """Result of a win probability prediction"""
    team1_name: str
    team2_name: str
    team1_win_prob: float
    team2_win_prob: float
    confidence: float  # 0-1, how confident the model is
    
    # Feature importances (which factors mattered most)
    top_factors: List[Tuple[str, float]] = None
    
    # Individual model predictions
    model_predictions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.top_factors is None:
            self.top_factors = []
        if self.model_predictions is None:
            self.model_predictions = {}
    
    @property
    def predicted_winner(self) -> str:
        return self.team1_name if self.team1_win_prob > 0.5 else self.team2_name
    
    @property
    def win_margin(self) -> float:
        """How decisive the prediction is (0.5 = toss-up, 1.0 = certain)"""
        return abs(self.team1_win_prob - 0.5) * 2


class SimplePredictor:
    """
    Rule-based predictor when sklearn isn't available
    Uses z-score differentials and tilt detection
    """
    
    def predict(self, features: MatchFeatures) -> float:
        """Predict team1 win probability"""
        # Base probability
        prob = 0.5
        
        # Z-score differential impact (~8% per unit)
        prob += features.z_score_diff * 0.08
        
        # K/D differential impact (~5% per 0.1 KD diff)  
        prob += features.kd_diff * 0.05
        
        # Tilt penalty (~5% per tilted player)
        prob -= features.team1_tilted_count * 0.05
        prob += features.team2_tilted_count * 0.05
        
        # Hot player bonus (~3% per hot player)
        prob += features.team1_hot_count * 0.03
        prob -= features.team2_hot_count * 0.03
        
        # Momentum impact (~5%)
        prob += features.momentum_diff * 0.05
        
        # Series score impact (team behind often plays harder)
        score_diff = features.team1_series_score - features.team2_series_score
        if score_diff > 0:
            prob += 0.03  # Slight advantage to leading team
        elif score_diff < 0:
            prob -= 0.03
        
        # Clamp to valid probability range
        return max(0.05, min(0.95, prob))


class EnsemblePredictor:
    """
    ML Ensemble for win probability prediction
    
    Usage:
        predictor = EnsemblePredictor(baseline_calculator)
        predictor.train(matches)  # Train on historical data
        result = predictor.predict_game(game_result)
    """
    
    # Ensemble weights
    WEIGHTS = {
        "logistic": 0.25,
        "random_forest": 0.35,
        "gradient_boost": 0.40
    }
    
    def __init__(self, calculator: BaselineCalculator):
        self.calculator = calculator
        self.extractor = FeatureExtractor(calculator)
        self.simple_predictor = SimplePredictor()
        
        self.is_trained = False
        self.scaler = None
        self.models = {}
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.models = {
                "logistic": LogisticRegression(max_iter=1000, random_state=42),
                "random_forest": RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=5,
                    random_state=42
                ),
                "gradient_boost": GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=42
                )
            }
    
    def train(self, matches: List) -> Dict[str, float]:
        """
        Train the ensemble on historical match data
        
        Args:
            matches: List of MatchResult objects
            
        Returns:
            Dict of model accuracies
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️ sklearn not available - using rule-based predictor")
            self.is_trained = True
            return {"simple": 1.0}
        
        # Extract features from all games
        X = []
        y = []
        
        for match in matches:
            features_list = self.extractor.extract_match_features(match)
            for features in features_list:
                X.append(features.to_vector())
                y.append(1 if features.team1_won else 0)
        
        if len(X) < 10:
            print(f"⚠️ Only {len(X)} samples - using rule-based predictor")
            self.is_trained = True
            return {"simple": 1.0}
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        accuracies = {}
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            accuracies[name] = model.score(X_scaled, y)
            print(f"   {name}: {accuracies[name]:.1%} accuracy")
        
        self.is_trained = True
        return accuracies
    
    def predict_game(
        self, 
        game_result,
        team1_series_score: int = 0,
        team2_series_score: int = 0,
        previous_games: List = None
    ) -> PredictionResult:
        """
        Predict win probability for a game
        
        Args:
            game_result: GameResult object
            team1_series_score: Current series score
            team2_series_score: Current series score
            previous_games: Previous games in series
            
        Returns:
            PredictionResult with probabilities
        """
        # Extract features
        features = self.extractor.extract_game_features(
            game_result,
            team1_series_score,
            team2_series_score,
            previous_games
        )
        
        return self.predict_from_features(
            features,
            game_result.team1_name,
            game_result.team2_name
        )
    
    def predict_from_features(
        self,
        features: MatchFeatures,
        team1_name: str,
        team2_name: str
    ) -> PredictionResult:
        """Predict from pre-extracted features"""
        
        model_predictions = {}
        
        if not SKLEARN_AVAILABLE or not self.is_trained or not self.models:
            # Use simple predictor
            prob = self.simple_predictor.predict(features)
            model_predictions["rule_based"] = prob
        else:
            # Get predictions from each model
            X = np.array([features.to_vector()])
            X_scaled = self.scaler.transform(X)
            
            for name, model in self.models.items():
                try:
                    prob = model.predict_proba(X_scaled)[0][1]
                    model_predictions[name] = prob
                except:
                    pass
        
        # Ensemble prediction (weighted average)
        if len(model_predictions) > 1:
            total_weight = 0
            weighted_prob = 0
            for name, prob in model_predictions.items():
                weight = self.WEIGHTS.get(name, 0.33)
                weighted_prob += prob * weight
                total_weight += weight
            team1_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        else:
            team1_prob = list(model_predictions.values())[0] if model_predictions else 0.5
        
        # Calculate confidence based on model agreement
        if len(model_predictions) > 1:
            probs = list(model_predictions.values())
            confidence = 1 - np.std(probs) * 2  # Lower std = higher confidence
            confidence = max(0.3, min(1.0, confidence))
        else:
            confidence = 0.7
        
        # Get top factors
        top_factors = self._get_top_factors(features)
        
        return PredictionResult(
            team1_name=team1_name,
            team2_name=team2_name,
            team1_win_prob=team1_prob,
            team2_win_prob=1 - team1_prob,
            confidence=confidence,
            top_factors=top_factors,
            model_predictions=model_predictions
        )
    
    def _get_top_factors(self, features: MatchFeatures) -> List[Tuple[str, float]]:
        """Identify the most important factors in the prediction"""
        factors = []
        
        # Z-score differential
        if abs(features.z_score_diff) > 0.5:
            direction = "favors" if features.z_score_diff > 0 else "hurts"
            factors.append((f"Team performance deviation {direction} Team 1", features.z_score_diff))
        
        # Tilt detection
        if features.team1_tilted_count > 0:
            factors.append((f"Team 1 has {features.team1_tilted_count} tilted player(s)", -0.05 * features.team1_tilted_count))
        if features.team2_tilted_count > 0:
            factors.append((f"Team 2 has {features.team2_tilted_count} tilted player(s)", 0.05 * features.team2_tilted_count))
        
        # Hot players
        if features.team1_hot_count > 0:
            factors.append((f"Team 1 has {features.team1_hot_count} hot player(s)", 0.03 * features.team1_hot_count))
        if features.team2_hot_count > 0:
            factors.append((f"Team 2 has {features.team2_hot_count} hot player(s)", -0.03 * features.team2_hot_count))
        
        # K/D differential
        if abs(features.kd_diff) > 0.2:
            better = "Team 1" if features.kd_diff > 0 else "Team 2"
            factors.append((f"{better} has better K/D ({features.kd_diff:+.2f})", features.kd_diff * 0.05))
        
        # Momentum
        if abs(features.momentum_diff) > 0.2:
            momentum_team = "Team 1" if features.momentum_diff > 0 else "Team 2"
            factors.append((f"{momentum_team} has momentum", features.momentum_diff * 0.05))
        
        # Sort by absolute impact
        factors.sort(key=lambda x: abs(x[1]), reverse=True)
        return factors[:5]  # Top 5 factors
    
    def evaluate(self, matches: List) -> Dict[str, float]:
        """
        Evaluate prediction accuracy on matches
        
        Returns accuracy metrics
        """
        correct = 0
        total = 0
        
        for match in matches:
            team1_score = 0
            team2_score = 0
            previous_games = []
            
            for game in match.games:
                result = self.predict_game(
                    game,
                    team1_score,
                    team2_score,
                    previous_games
                )
                
                actual_winner = game.winner
                predicted_winner = result.predicted_winner
                
                if predicted_winner == actual_winner:
                    correct += 1
                total += 1
                
                # Update for next game
                if game.winner == match.team1_name:
                    team1_score += 1
                else:
                    team2_score += 1
                previous_games.append(game)
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }
