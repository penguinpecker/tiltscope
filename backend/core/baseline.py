"""
Player Baseline Calculator
Computes historical performance baselines for tilt detection

This module builds a statistical profile for each player:
- Mean K/D ratio
- Standard deviation
- Performance trends
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics


@dataclass
class PlayerBaseline:
    """Statistical baseline for a player's performance"""
    player_name: str
    team_name: str
    
    # Core stats
    games_played: int = 0
    total_kills: int = 0
    total_deaths: int = 0
    total_assists: int = 0
    
    # K/D history for variance calculation
    kd_history: List[float] = field(default_factory=list)
    
    # Computed stats (call compute() to populate)
    kd_mean: float = 0.0
    kd_std_dev: float = 0.0
    kd_min: float = 0.0
    kd_max: float = 0.0
    
    # Role/agent stats
    agents_played: Dict[str, int] = field(default_factory=dict)
    most_played_agent: str = ""
    
    def add_game(self, kills: int, deaths: int, assists: int, agent: str = ""):
        """Add a game to the player's history"""
        self.games_played += 1
        self.total_kills += kills
        self.total_deaths += deaths
        self.total_assists += assists
        
        # Calculate K/D for this game
        kd = kills / max(deaths, 1)
        self.kd_history.append(kd)
        
        # Track agent usage
        if agent:
            self.agents_played[agent] = self.agents_played.get(agent, 0) + 1
    
    def compute(self):
        """Compute statistical measures from history"""
        if not self.kd_history:
            return
        
        self.kd_mean = statistics.mean(self.kd_history)
        self.kd_min = min(self.kd_history)
        self.kd_max = max(self.kd_history)
        
        if len(self.kd_history) >= 2:
            self.kd_std_dev = statistics.stdev(self.kd_history)
        else:
            self.kd_std_dev = 0.2  # Default std dev if not enough data
        
        # Find most played agent
        if self.agents_played:
            self.most_played_agent = max(self.agents_played, key=self.agents_played.get)
    
    @property
    def overall_kd(self) -> float:
        """Overall K/D across all games"""
        return self.total_kills / max(self.total_deaths, 1)
    
    @property
    def overall_kda(self) -> float:
        """Overall KDA across all games"""
        return (self.total_kills + self.total_assists) / max(self.total_deaths, 1)
    
    def __repr__(self):
        return (f"PlayerBaseline({self.player_name}: {self.games_played} games, "
                f"KD={self.kd_mean:.2f}±{self.kd_std_dev:.2f})")


class BaselineCalculator:
    """
    Builds player baselines from match history
    
    Usage:
        calculator = BaselineCalculator()
        calculator.add_matches(matches)  # List of MatchResult
        baseline = calculator.get_baseline("OXY")
    """
    
    def __init__(self):
        self.player_baselines: Dict[str, PlayerBaseline] = {}
    
    def add_matches(self, matches: List) -> None:
        """
        Process a list of MatchResult objects to build baselines
        
        Args:
            matches: List of MatchResult from GRIDClient
        """
        for match in matches:
            for game in match.games:
                for player_stats in game.player_stats:
                    self._add_player_game(
                        player_name=player_stats.player_name,
                        team_name=player_stats.team_name,
                        kills=player_stats.kills,
                        deaths=player_stats.deaths,
                        assists=player_stats.assists,
                        agent=player_stats.agent_or_champion
                    )
        
        # Compute statistics for all players
        for baseline in self.player_baselines.values():
            baseline.compute()
    
    def _add_player_game(self, player_name: str, team_name: str, 
                         kills: int, deaths: int, assists: int, agent: str = ""):
        """Add a single game for a player"""
        key = player_name.lower()
        
        if key not in self.player_baselines:
            self.player_baselines[key] = PlayerBaseline(
                player_name=player_name,
                team_name=team_name
            )
        
        self.player_baselines[key].add_game(kills, deaths, assists, agent)
    
    def get_baseline(self, player_name: str) -> Optional[PlayerBaseline]:
        """Get baseline for a specific player"""
        return self.player_baselines.get(player_name.lower())
    
    def get_team_baselines(self, team_name: str) -> List[PlayerBaseline]:
        """Get all baselines for players on a team"""
        return [
            b for b in self.player_baselines.values()
            if team_name.lower() in b.team_name.lower()
        ]
    
    def get_all_baselines(self) -> List[PlayerBaseline]:
        """Get all player baselines"""
        return list(self.player_baselines.values())
    
    def print_summary(self):
        """Print a summary of all baselines"""
        print("\n" + "=" * 70)
        print("📊 PLAYER BASELINES")
        print("=" * 70)
        
        # Sort by K/D mean descending
        sorted_baselines = sorted(
            self.player_baselines.values(),
            key=lambda b: b.kd_mean,
            reverse=True
        )
        
        print(f"{'Player':<12} {'Team':<12} {'Games':>6} {'K/D Mean':>10} {'Std Dev':>10} {'Range':>15}")
        print("-" * 70)
        
        for b in sorted_baselines:
            range_str = f"{b.kd_min:.2f}-{b.kd_max:.2f}"
            print(f"{b.player_name:<12} {b.team_name[:11]:<12} {b.games_played:>6} "
                  f"{b.kd_mean:>10.2f} {b.kd_std_dev:>10.2f} {range_str:>15}")
