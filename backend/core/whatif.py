"""
TiltScope What-If Monte Carlo Engine
=====================================
Simulates alternative scenarios to answer questions like:
- "What if OXY wasn't tilted in Game 2?"
- "What if NRG's mada didn't go off?"
- "How would the match change with different player performance?"

Uses Monte Carlo simulation to estimate probability distributions.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import statistics

from .baseline import BaselineCalculator, PlayerBaseline
from .deviation import DeviationEngine, PerformanceState, PlayerDeviation
from .features import FeatureExtractor, MatchFeatures
from .predictor import EnsemblePredictor, PredictionResult


@dataclass
class WhatIfScenario:
    """A hypothetical scenario to simulate"""
    description: str
    player_adjustments: Dict[str, float]  # player_name -> z-score adjustment
    
    def __repr__(self):
        return f"WhatIfScenario('{self.description}')"


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation"""
    scenario: WhatIfScenario
    
    # Original prediction
    original_team1_prob: float
    original_team2_prob: float
    
    # Simulated prediction (average over iterations)
    simulated_team1_prob: float
    simulated_team2_prob: float
    
    # Probability distribution from simulations
    team1_prob_std: float
    team1_prob_min: float
    team1_prob_max: float
    
    # Impact analysis
    probability_shift: float  # How much team1's prob changed
    outcome_changed: bool  # Would the predicted winner change?
    
    # Simulation details
    num_simulations: int
    
    @property
    def impact_description(self) -> str:
        """Human-readable impact description"""
        shift_pct = self.probability_shift * 100
        if abs(shift_pct) < 2:
            return "Minimal impact"
        elif abs(shift_pct) < 5:
            direction = "increase" if shift_pct > 0 else "decrease"
            return f"Slight {direction} ({shift_pct:+.1f}%)"
        elif abs(shift_pct) < 10:
            direction = "increase" if shift_pct > 0 else "decrease"
            return f"Moderate {direction} ({shift_pct:+.1f}%)"
        else:
            direction = "increase" if shift_pct > 0 else "decrease"
            return f"Significant {direction} ({shift_pct:+.1f}%)"


class WhatIfEngine:
    """
    Monte Carlo simulation engine for what-if analysis
    
    Usage:
        engine = WhatIfEngine(predictor)
        
        # Create a scenario
        scenario = engine.create_scenario_fix_tilt("OXY", "Cloud9")
        
        # Run simulation
        result = engine.simulate(game_result, scenario)
        
        print(f"If {scenario.description}:")
        print(f"  Win probability would change by {result.probability_shift:+.1%}")
    """
    
    DEFAULT_SIMULATIONS = 100
    
    def __init__(self, predictor: EnsemblePredictor):
        self.predictor = predictor
        self.calculator = predictor.calculator
        self.engine = DeviationEngine(self.calculator)
    
    # =========================================================================
    # SCENARIO BUILDERS
    # =========================================================================
    
    def create_scenario_fix_tilt(self, player_name: str, team_name: str) -> WhatIfScenario:
        """
        Create scenario: What if this tilted player performed at baseline?
        """
        return WhatIfScenario(
            description=f"{player_name} performed at their baseline",
            player_adjustments={player_name: 0.0}  # Reset to z=0 (baseline)
        )
    
    def create_scenario_player_on_fire(self, player_name: str, team_name: str) -> WhatIfScenario:
        """
        Create scenario: What if this player was on fire? (z=+2.0)
        """
        return WhatIfScenario(
            description=f"{player_name} was on fire",
            player_adjustments={player_name: 2.0}  # Set to z=+2.0
        )
    
    def create_scenario_player_tilted(self, player_name: str, team_name: str) -> WhatIfScenario:
        """
        Create scenario: What if this player was tilted? (z=-2.0)
        """
        return WhatIfScenario(
            description=f"{player_name} was tilted",
            player_adjustments={player_name: -2.0}  # Set to z=-2.0
        )
    
    def create_scenario_swap_performance(
        self, 
        player1_name: str, 
        player2_name: str
    ) -> WhatIfScenario:
        """
        Create scenario: What if two players swapped performance levels?
        """
        return WhatIfScenario(
            description=f"{player1_name} and {player2_name} swapped performance",
            player_adjustments={
                player1_name: "swap",  # Will be handled specially
                player2_name: "swap"
            }
        )
    
    def create_scenario_team_baseline(self, team_name: str) -> WhatIfScenario:
        """
        Create scenario: What if entire team performed at baseline?
        """
        return WhatIfScenario(
            description=f"{team_name} all performed at baseline",
            player_adjustments={f"TEAM:{team_name}": 0.0}
        )
    
    def auto_generate_scenarios(self, game_result) -> List[WhatIfScenario]:
        """
        Automatically generate interesting scenarios based on game data
        """
        scenarios = []
        
        # Analyze the game
        team1_dev, team2_dev = self.engine.analyze_game(game_result)
        
        # Scenario for each tilted player: "What if they weren't tilted?"
        for dev in [team1_dev, team2_dev]:
            for player in dev.tilted_players:
                scenarios.append(self.create_scenario_fix_tilt(
                    player.player_name, 
                    player.team_name
                ))
        
        # Scenario for best performer: "What if they had a normal game?"
        all_players = team1_dev.player_deviations + team2_dev.player_deviations
        if all_players:
            best_player = max(all_players, key=lambda p: p.z_score)
            if best_player.z_score > 1.0:
                scenarios.append(WhatIfScenario(
                    description=f"{best_player.player_name} had an average game instead",
                    player_adjustments={best_player.player_name: 0.0}
                ))
        
        # Team-level scenarios
        scenarios.append(self.create_scenario_team_baseline(game_result.team1_name))
        scenarios.append(self.create_scenario_team_baseline(game_result.team2_name))
        
        return scenarios
    
    # =========================================================================
    # SIMULATION
    # =========================================================================
    
    def simulate(
        self,
        game_result,
        scenario: WhatIfScenario,
        num_simulations: int = DEFAULT_SIMULATIONS,
        team1_series_score: int = 0,
        team2_series_score: int = 0,
        previous_games: List = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a scenario
        
        Args:
            game_result: Original game result
            scenario: What-if scenario to simulate
            num_simulations: Number of simulation iterations
            team1_series_score: Current series score
            team2_series_score: Current series score
            previous_games: Previous games in series
            
        Returns:
            SimulationResult with probability distributions
        """
        # Get original prediction
        original_result = self.predictor.predict_game(
            game_result,
            team1_series_score,
            team2_series_score,
            previous_games
        )
        
        # Run simulations
        simulated_probs = []
        
        for _ in range(num_simulations):
            # Create modified features based on scenario
            modified_features = self._apply_scenario(
                game_result,
                scenario,
                team1_series_score,
                team2_series_score,
                previous_games
            )
            
            # Add some randomness to simulate variance
            modified_features = self._add_simulation_noise(modified_features)
            
            # Get prediction for modified scenario
            pred = self.predictor.predict_from_features(
                modified_features,
                game_result.team1_name,
                game_result.team2_name
            )
            simulated_probs.append(pred.team1_win_prob)
        
        # Calculate statistics
        avg_prob = statistics.mean(simulated_probs)
        std_prob = statistics.stdev(simulated_probs) if len(simulated_probs) > 1 else 0
        
        return SimulationResult(
            scenario=scenario,
            original_team1_prob=original_result.team1_win_prob,
            original_team2_prob=original_result.team2_win_prob,
            simulated_team1_prob=avg_prob,
            simulated_team2_prob=1 - avg_prob,
            team1_prob_std=std_prob,
            team1_prob_min=min(simulated_probs),
            team1_prob_max=max(simulated_probs),
            probability_shift=avg_prob - original_result.team1_win_prob,
            outcome_changed=(avg_prob > 0.5) != (original_result.team1_win_prob > 0.5),
            num_simulations=num_simulations
        )
    
    def _apply_scenario(
        self,
        game_result,
        scenario: WhatIfScenario,
        team1_series_score: int,
        team2_series_score: int,
        previous_games: List
    ) -> MatchFeatures:
        """Apply scenario adjustments to create modified features"""
        
        # Get original features
        features = self.predictor.extractor.extract_game_features(
            game_result,
            team1_series_score,
            team2_series_score,
            previous_games
        )
        
        # Get original deviations
        team1_dev, team2_dev = self.engine.analyze_game(game_result)
        
        # Apply adjustments
        for player_key, adjustment in scenario.player_adjustments.items():
            if player_key.startswith("TEAM:"):
                # Team-wide adjustment
                team_name = player_key.replace("TEAM:", "")
                if team_name.lower() in game_result.team1_name.lower():
                    features.team1_avg_z_score = adjustment
                    features.team1_tilted_count = 0
                    features.team1_hot_count = 0 if adjustment == 0 else features.team1_hot_count
                else:
                    features.team2_avg_z_score = adjustment
                    features.team2_tilted_count = 0
                    features.team2_hot_count = 0 if adjustment == 0 else features.team2_hot_count
            else:
                # Individual player adjustment
                self._adjust_player_in_features(
                    features, 
                    player_key, 
                    adjustment,
                    team1_dev,
                    team2_dev,
                    game_result
                )
        
        # Recalculate differentials
        features.z_score_diff = features.team1_avg_z_score - features.team2_avg_z_score
        
        return features
    
    def _adjust_player_in_features(
        self,
        features: MatchFeatures,
        player_name: str,
        target_z: float,
        team1_dev,
        team2_dev,
        game_result
    ):
        """Adjust features for a single player's performance change"""
        
        # Find which team the player is on
        player_dev = None
        is_team1 = False
        
        for p in team1_dev.player_deviations:
            if p.player_name.lower() == player_name.lower():
                player_dev = p
                is_team1 = True
                break
        
        if not player_dev:
            for p in team2_dev.player_deviations:
                if p.player_name.lower() == player_name.lower():
                    player_dev = p
                    is_team1 = False
                    break
        
        if not player_dev:
            return  # Player not found
        
        # Calculate z-score change
        old_z = player_dev.z_score
        z_change = target_z - old_z
        
        # Update team averages (simplified - assumes 5 players)
        z_impact = z_change / 5
        
        if is_team1:
            features.team1_avg_z_score += z_impact
            
            # Update tilt/hot counts
            old_state = player_dev.state
            new_state = DeviationEngine._z_to_state(target_z)
            
            if old_state == PerformanceState.TILTED and new_state != PerformanceState.TILTED:
                features.team1_tilted_count = max(0, features.team1_tilted_count - 1)
            elif old_state != PerformanceState.TILTED and new_state == PerformanceState.TILTED:
                features.team1_tilted_count += 1
            
            if old_state in [PerformanceState.HOT, PerformanceState.ON_FIRE]:
                if new_state not in [PerformanceState.HOT, PerformanceState.ON_FIRE]:
                    features.team1_hot_count = max(0, features.team1_hot_count - 1)
            elif new_state in [PerformanceState.HOT, PerformanceState.ON_FIRE]:
                features.team1_hot_count += 1
        else:
            features.team2_avg_z_score += z_impact
            
            # Update tilt/hot counts (similar logic)
            old_state = player_dev.state
            new_state = DeviationEngine._z_to_state(target_z)
            
            if old_state == PerformanceState.TILTED and new_state != PerformanceState.TILTED:
                features.team2_tilted_count = max(0, features.team2_tilted_count - 1)
            elif old_state != PerformanceState.TILTED and new_state == PerformanceState.TILTED:
                features.team2_tilted_count += 1
            
            if old_state in [PerformanceState.HOT, PerformanceState.ON_FIRE]:
                if new_state not in [PerformanceState.HOT, PerformanceState.ON_FIRE]:
                    features.team2_hot_count = max(0, features.team2_hot_count - 1)
            elif new_state in [PerformanceState.HOT, PerformanceState.ON_FIRE]:
                features.team2_hot_count += 1
    
    def _add_simulation_noise(self, features: MatchFeatures) -> MatchFeatures:
        """Add random noise to simulate variance"""
        # Small random adjustments to simulate real-world variance
        noise_scale = 0.1
        
        features.team1_avg_z_score += random.gauss(0, noise_scale)
        features.team2_avg_z_score += random.gauss(0, noise_scale)
        features.z_score_diff = features.team1_avg_z_score - features.team2_avg_z_score
        
        return features
    
    # =========================================================================
    # ANALYSIS HELPERS
    # =========================================================================
    
    def run_all_scenarios(
        self,
        game_result,
        team1_series_score: int = 0,
        team2_series_score: int = 0,
        previous_games: List = None
    ) -> List[SimulationResult]:
        """
        Run all auto-generated scenarios and return sorted by impact
        """
        scenarios = self.auto_generate_scenarios(game_result)
        
        results = []
        for scenario in scenarios:
            result = self.simulate(
                game_result,
                scenario,
                num_simulations=50,  # Fewer iterations for speed
                team1_series_score=team1_series_score,
                team2_series_score=team2_series_score,
                previous_games=previous_games
            )
            results.append(result)
        
        # Sort by absolute probability shift
        results.sort(key=lambda r: abs(r.probability_shift), reverse=True)
        
        return results
    
    def print_analysis(
        self,
        game_result,
        team1_series_score: int = 0,
        team2_series_score: int = 0,
        previous_games: List = None
    ):
        """Print a formatted what-if analysis"""
        
        print(f"\n{'='*70}")
        print(f"🔮 WHAT-IF ANALYSIS: {game_result.team1_name} vs {game_result.team2_name}")
        print(f"   Map: {game_result.map_name.upper()} | Score: {game_result.team1_score}-{game_result.team2_score}")
        print(f"{'='*70}")
        
        results = self.run_all_scenarios(
            game_result,
            team1_series_score,
            team2_series_score,
            previous_games
        )
        
        # Original prediction
        original = self.predictor.predict_game(
            game_result,
            team1_series_score,
            team2_series_score,
            previous_games
        )
        
        print(f"\n📊 Original Prediction:")
        print(f"   {game_result.team1_name}: {original.team1_win_prob:.1%}")
        print(f"   {game_result.team2_name}: {original.team2_win_prob:.1%}")
        
        print(f"\n🔮 What-If Scenarios (sorted by impact):")
        print("-" * 70)
        
        for i, result in enumerate(results[:5]):  # Top 5
            outcome_marker = "⚡" if result.outcome_changed else ""
            
            print(f"\n   {i+1}. What if {result.scenario.description}?")
            print(f"      {game_result.team1_name}: {result.original_team1_prob:.1%} → {result.simulated_team1_prob:.1%} ({result.probability_shift:+.1%}) {outcome_marker}")
            print(f"      Impact: {result.impact_description}")
            
            if result.outcome_changed:
                print(f"      ⚡ This would CHANGE the predicted winner!")
        
        print(f"\n{'='*70}")
