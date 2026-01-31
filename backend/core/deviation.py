"""
TILTSCOPE - Deviation Engine
============================
THE CORE INNOVATION: Real-time performance deviation detection

This module detects when players are:
- TILTED: Performing significantly below baseline (z < -1.5)
- COLD: Slightly below baseline (-1.5 < z < -0.5)
- NORMAL: Within expected range (-0.5 < z < 0.5)
- HOT: Slightly above baseline (0.5 < z < 1.5)  
- ON FIRE: Performing significantly above baseline (z > 1.5)

The z-score measures how many standard deviations away from
the mean a player's current performance is.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from .baseline import BaselineCalculator, PlayerBaseline


class PerformanceState(Enum):
    """Player performance state based on deviation from baseline"""
    ON_FIRE = "🔥 ON FIRE"      # z > 1.5
    HOT = "📈 Hot"              # 0.5 < z < 1.5
    NORMAL = "➖ Normal"         # -0.5 < z < 0.5
    COLD = "📉 Cold"            # -1.5 < z < -0.5
    TILTED = "💀 TILTED"        # z < -1.5


@dataclass
class PlayerDeviation:
    """Deviation analysis for a single player in a game"""
    player_name: str
    team_name: str
    agent: str
    
    # Current game stats
    current_kills: int
    current_deaths: int
    current_kd: float
    
    # Baseline comparison
    baseline_kd_mean: float
    baseline_kd_std: float
    
    # Deviation metrics
    z_score: float
    state: PerformanceState
    
    # Impact assessment
    win_probability_impact: float  # How much this affects team win %
    
    @property
    def deviation_percentage(self) -> float:
        """How far off baseline as a percentage"""
        if self.baseline_kd_mean == 0:
            return 0.0
        return ((self.current_kd - self.baseline_kd_mean) / self.baseline_kd_mean) * 100
    
    def __repr__(self):
        return (f"{self.player_name} [{self.agent}]: {self.current_kd:.2f} KD "
                f"(baseline: {self.baseline_kd_mean:.2f}) → {self.state.value} (z={self.z_score:+.2f})")


@dataclass
class TeamDeviation:
    """Aggregated deviation for an entire team"""
    team_name: str
    player_deviations: List[PlayerDeviation]
    
    @property
    def average_z_score(self) -> float:
        """Average z-score across all players"""
        if not self.player_deviations:
            return 0.0
        return sum(p.z_score for p in self.player_deviations) / len(self.player_deviations)
    
    @property
    def tilted_players(self) -> List[PlayerDeviation]:
        """Players who are tilted"""
        return [p for p in self.player_deviations if p.state == PerformanceState.TILTED]
    
    @property
    def hot_players(self) -> List[PlayerDeviation]:
        """Players who are on fire"""
        return [p for p in self.player_deviations if p.state in [PerformanceState.ON_FIRE, PerformanceState.HOT]]
    
    @property
    def team_state(self) -> PerformanceState:
        """Overall team performance state"""
        avg_z = self.average_z_score
        return DeviationEngine._z_to_state(avg_z)
    
    @property
    def total_win_impact(self) -> float:
        """Combined win probability impact"""
        return sum(p.win_probability_impact for p in self.player_deviations)


class DeviationEngine:
    """
    Core TiltScope engine - detects performance deviations
    
    Usage:
        engine = DeviationEngine(baseline_calculator)
        deviations = engine.analyze_game(game_result)
        
        for team in deviations:
            print(f"{team.team_name}: {team.team_state.value}")
            for player in team.tilted_players:
                print(f"  ⚠️ {player}")
    """
    
    # Z-score thresholds for performance states
    Z_ON_FIRE = 1.5
    Z_HOT = 0.5
    Z_COLD = -0.5
    Z_TILTED = -1.5
    
    # Win probability impact per z-score unit (estimated)
    WIN_IMPACT_PER_Z = 0.03  # 3% per standard deviation
    
    def __init__(self, calculator: BaselineCalculator):
        self.calculator = calculator
    
    @staticmethod
    def _z_to_state(z: float) -> PerformanceState:
        """Convert z-score to performance state"""
        if z >= DeviationEngine.Z_ON_FIRE:
            return PerformanceState.ON_FIRE
        elif z >= DeviationEngine.Z_HOT:
            return PerformanceState.HOT
        elif z >= DeviationEngine.Z_COLD:
            return PerformanceState.NORMAL
        elif z >= DeviationEngine.Z_TILTED:
            return PerformanceState.COLD
        else:
            return PerformanceState.TILTED
    
    def calculate_z_score(self, current_kd: float, baseline: PlayerBaseline) -> float:
        """
        Calculate z-score for current performance vs baseline
        
        z = (x - μ) / σ
        
        Where:
        - x = current K/D
        - μ = baseline mean K/D
        - σ = baseline std dev
        """
        if baseline.kd_std_dev == 0:
            # If no variance, use a default std dev
            std = 0.2
        else:
            std = baseline.kd_std_dev
        
        return (current_kd - baseline.kd_mean) / std
    
    def analyze_player(self, player_name: str, team_name: str, agent: str,
                       kills: int, deaths: int) -> Optional[PlayerDeviation]:
        """
        Analyze a single player's deviation from baseline
        
        Args:
            player_name: Player's name
            team_name: Team name
            agent: Agent/champion played
            kills: Current game kills
            deaths: Current game deaths
            
        Returns:
            PlayerDeviation object or None if no baseline exists
        """
        baseline = self.calculator.get_baseline(player_name)
        
        if not baseline:
            # No baseline - can't calculate deviation
            return None
        
        current_kd = kills / max(deaths, 1)
        z_score = self.calculate_z_score(current_kd, baseline)
        state = self._z_to_state(z_score)
        
        # Calculate win probability impact
        win_impact = z_score * self.WIN_IMPACT_PER_Z
        
        return PlayerDeviation(
            player_name=player_name,
            team_name=team_name,
            agent=agent,
            current_kills=kills,
            current_deaths=deaths,
            current_kd=current_kd,
            baseline_kd_mean=baseline.kd_mean,
            baseline_kd_std=baseline.kd_std_dev,
            z_score=z_score,
            state=state,
            win_probability_impact=win_impact
        )
    
    def analyze_game(self, game_result) -> Tuple[TeamDeviation, TeamDeviation]:
        """
        Analyze all players in a game result
        
        Args:
            game_result: GameResult object from grid_client
            
        Returns:
            Tuple of (team1_deviation, team2_deviation)
        """
        # Group players by team
        team_players: Dict[str, List[PlayerDeviation]] = {}
        
        for player_stats in game_result.player_stats:
            deviation = self.analyze_player(
                player_name=player_stats.player_name,
                team_name=player_stats.team_name,
                agent=player_stats.agent_or_champion,
                kills=player_stats.kills,
                deaths=player_stats.deaths
            )
            
            if deviation:
                team = player_stats.team_name
                if team not in team_players:
                    team_players[team] = []
                team_players[team].append(deviation)
        
        # Create TeamDeviation objects
        teams = list(team_players.keys())
        
        team1 = TeamDeviation(
            team_name=teams[0] if teams else "Unknown",
            player_deviations=team_players.get(teams[0], []) if teams else []
        )
        
        team2 = TeamDeviation(
            team_name=teams[1] if len(teams) > 1 else "Unknown",
            player_deviations=team_players.get(teams[1], []) if len(teams) > 1 else []
        )
        
        return team1, team2
    
    def print_game_analysis(self, game_result, game_num: int = 1):
        """Print a formatted analysis of a game"""
        team1_dev, team2_dev = self.analyze_game(game_result)
        
        print(f"\n{'='*70}")
        print(f"🎮 GAME {game_num}: {game_result.map_name.upper()}")
        print(f"   {game_result.team1_name} {game_result.team1_score} - {game_result.team2_score} {game_result.team2_name}")
        print(f"{'='*70}")
        
        for team_dev in [team1_dev, team2_dev]:
            # Team header with state
            print(f"\n🏆 {team_dev.team_name} {team_dev.team_state.value}")
            print(f"   Team Z-Score: {team_dev.average_z_score:+.2f} | Win Impact: {team_dev.total_win_impact:+.1%}")
            print("-" * 50)
            
            # Sort by z-score (worst performers first)
            sorted_players = sorted(team_dev.player_deviations, key=lambda p: p.z_score)
            
            for p in sorted_players:
                # State emoji
                state_str = p.state.value
                
                # Color-code the K/D based on performance
                kd_str = f"{p.current_kd:.2f}"
                baseline_str = f"{p.baseline_kd_mean:.2f}"
                
                print(f"   {p.player_name:<10} [{p.agent:<10}] "
                      f"KD: {kd_str} (baseline: {baseline_str}) "
                      f"z={p.z_score:+.2f} {state_str}")
        
        # Insights
        print(f"\n💡 INSIGHTS:")
        
        all_tilted = team1_dev.tilted_players + team2_dev.tilted_players
        all_hot = team1_dev.hot_players + team2_dev.hot_players
        
        if all_tilted:
            print(f"   ⚠️ TILTED PLAYERS:")
            for p in all_tilted:
                print(f"      • {p.player_name} ({p.team_name}): {p.deviation_percentage:+.0f}% below baseline")
        
        if all_hot:
            print(f"   🔥 HOT PLAYERS:")
            for p in all_hot:
                print(f"      • {p.player_name} ({p.team_name}): {p.deviation_percentage:+.0f}% above baseline")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Run test_deviation.py for full test")
