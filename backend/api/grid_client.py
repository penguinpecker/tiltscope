"""
GRID API Client - WORKING VERSION
Fetches REAL esports data for VALORANT and League of Legends
"""

import os
import asyncio
from typing import Optional, List, Dict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import httpx
from dotenv import load_dotenv

load_dotenv()


class GameType(Enum):
    VALORANT = "val"
    LEAGUE_OF_LEGENDS = "lol"


@dataclass
class PlayerGameStats:
    """Player stats for a single game/map"""
    player_name: str
    team_name: str
    agent_or_champion: str
    kills: int
    deaths: int
    assists: int
    first_kill: bool = False
    
    @property
    def kd_ratio(self) -> float:
        return self.kills / max(self.deaths, 1)
    
    @property
    def kda(self) -> float:
        return (self.kills + self.assists) / max(self.deaths, 1)


@dataclass
class GameResult:
    """Single game/map result"""
    game_number: int
    map_name: str
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    winner: str
    player_stats: List[PlayerGameStats] = field(default_factory=list)


@dataclass 
class MatchResult:
    """Complete match (series) result"""
    series_id: str
    game_type: GameType
    date: datetime
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    winner: str
    tournament_name: str
    games: List[GameResult] = field(default_factory=list)


class GRIDClient:
    TITLE_IDS = {
        GameType.VALORANT: "6",
        GameType.LEAGUE_OF_LEGENDS: "3"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GRID_API_KEY")
        if not self.api_key:
            raise ValueError("GRID API key required")
        
        self.central_data_url = "https://api-op.grid.gg/central-data/graphql"
        self.series_state_url = "https://api-op.grid.gg/live-data-feed/series-state/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
    
    async def _query(self, url: str, query: str, variables: Dict = None) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=self.headers,
                json={"query": query, "variables": variables or {}}
            )
            response.raise_for_status()
            result = response.json()
            if "errors" in result:
                raise Exception(f"GraphQL Error: {result['errors']}")
            return result.get("data", {})
    
    async def test_connection(self) -> bool:
        try:
            await self._query(self.central_data_url, "query { __schema { queryType { name } } }")
            return True
        except:
            return False
    
    async def get_recent_series(self, game_type: GameType, num_matches: int = 20, team_name: Optional[str] = None) -> List[Dict]:
        """Get recent series from Central Data API"""
        title_id = self.TITLE_IDS[game_type]
        
        query = """
        query RecentSeries($titleId: ID!, $first: Int!) {
            allSeries(
                filter: { titleId: $titleId }
                first: $first
                orderBy: StartTimeScheduled
                orderDirection: DESC
            ) {
                edges {
                    node {
                        id
                        startTimeScheduled
                        format { name }
                        tournament { name }
                        teams {
                            baseInfo { id name }
                        }
                    }
                }
            }
        }
        """
        
        result = await self._query(self.central_data_url, query, {"titleId": title_id, "first": num_matches * 2})
        
        series_list = []
        for edge in result.get("allSeries", {}).get("edges", []):
            node = edge["node"]
            teams = node.get("teams", [])
            
            if len(teams) < 2:
                continue
            
            team1_name = teams[0].get("baseInfo", {}).get("name", "Unknown")
            team2_name = teams[1].get("baseInfo", {}).get("name", "Unknown")
            
            if team_name:
                if not (team_name.lower() in team1_name.lower() or team_name.lower() in team2_name.lower()):
                    continue
            
            series_list.append({
                "id": node["id"],
                "date": node.get("startTimeScheduled"),
                "tournament": node.get("tournament", {}).get("name", "Unknown") if node.get("tournament") else "Unknown",
                "team1_name": team1_name,
                "team2_name": team2_name
            })
            
            if len(series_list) >= num_matches:
                break
        
        return series_list
    
    async def get_series_details(self, series_id: str) -> Optional[Dict]:
        """Get detailed game-by-game stats from Series State API"""
        query = """
        query GetSeries($seriesId: ID!) {
            seriesState(id: $seriesId) {
                id
                teams {
                    name
                    score
                    won
                }
                games {
                    sequenceNumber
                    map { name }
                    teams {
                        name
                        score
                        won
                        players {
                            name
                            character { name }
                            kills
                            deaths
                            killAssistsGiven
                            firstKill
                        }
                    }
                }
            }
        }
        """
        try:
            result = await self._query(self.series_state_url, query, {"seriesId": series_id})
            return result.get("seriesState")
        except Exception as e:
            print(f"  ⚠️ Error fetching {series_id}: {e}")
            return None
    
    async def get_team_matches(self, team_name: str, game_type: GameType, num_matches: int = 10) -> List[MatchResult]:
        """Main method: Get matches for a team with full game-by-game details"""
        print(f"🔍 Searching for: {team_name} ({game_type.value})")
        
        series_list = await self.get_recent_series(game_type, num_matches, team_name)
        
        if not series_list:
            print(f"❌ No matches found for '{team_name}'")
            return []
        
        print(f"✅ Found {len(series_list)} series")
        
        matches = []
        for i, series in enumerate(series_list):
            print(f"  [{i+1}/{len(series_list)}] {series['team1_name']} vs {series['team2_name']}")
            
            details = await self.get_series_details(series["id"])
            if details:
                match = self._parse_match(series, details, game_type)
                if match:
                    matches.append(match)
            
            await asyncio.sleep(0.1)
        
        print(f"✅ Retrieved {len(matches)} matches with game details")
        return matches
    
    def _parse_match(self, series: Dict, details: Dict, game_type: GameType) -> Optional[MatchResult]:
        try:
            series_teams = details.get("teams", [])
            if len(series_teams) < 2:
                return None
            
            # Parse each game
            games = []
            for game_data in details.get("games", []):
                game_teams = game_data.get("teams", [])
                if len(game_teams) < 2:
                    continue
                
                player_stats = []
                for team in game_teams:
                    team_name = team.get("name", "Unknown")
                    for player in team.get("players", []):
                        char = player.get("character", {})
                        player_stats.append(PlayerGameStats(
                            player_name=player.get("name", "Unknown"),
                            team_name=team_name,
                            agent_or_champion=char.get("name", "Unknown") if char else "Unknown",
                            kills=player.get("kills", 0) or 0,
                            deaths=player.get("deaths", 0) or 0,
                            assists=player.get("killAssistsGiven", 0) or 0,
                            first_kill=player.get("firstKill", False) or False
                        ))
                
                winner = game_teams[0]["name"] if game_teams[0].get("won") else game_teams[1]["name"]
                
                games.append(GameResult(
                    game_number=game_data.get("sequenceNumber", 1),
                    map_name=game_data.get("map", {}).get("name", "Unknown") if game_data.get("map") else "Unknown",
                    team1_name=game_teams[0].get("name", "Unknown"),
                    team2_name=game_teams[1].get("name", "Unknown"),
                    team1_score=game_teams[0].get("score", 0) or 0,
                    team2_score=game_teams[1].get("score", 0) or 0,
                    winner=winner,
                    player_stats=player_stats
                ))
            
            series_winner = series_teams[0]["name"] if series_teams[0].get("won") else series_teams[1]["name"]
            
            return MatchResult(
                series_id=series["id"],
                game_type=game_type,
                date=datetime.fromisoformat(series["date"].replace("Z", "+00:00")) if series.get("date") else datetime.now(),
                team1_name=series_teams[0].get("name", "Unknown"),
                team2_name=series_teams[1].get("name", "Unknown"),
                team1_score=series_teams[0].get("score", 0) or 0,
                team2_score=series_teams[1].get("score", 0) or 0,
                winner=series_winner,
                tournament_name=series.get("tournament", "Unknown"),
                games=games
            )
        except Exception as e:
            print(f"  ⚠️ Parse error: {e}")
            return None


# Test
if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("🎯 TILTSCOPE - GRID API TEST")
        print("=" * 60)
        
        client = GRIDClient()
        
        print("\n📊 Fetching Cloud9 VALORANT matches...")
        matches = await client.get_team_matches("Cloud9", GameType.VALORANT, 2)
        
        for match in matches:
            print(f"\n{'='*60}")
            print(f"📍 {match.team1_name} vs {match.team2_name}")
            print(f"   Series: {match.team1_score}-{match.team2_score} | Winner: {match.winner}")
            print(f"   Tournament: {match.tournament_name}")
            
            for game in match.games:
                print(f"\n   🗺️  Game {game.game_number}: {game.map_name.upper()}")
                print(f"      {game.team1_name} {game.team1_score} - {game.team2_score} {game.team2_name}")
                print(f"      Players:")
                for p in game.player_stats[:5]:
                    kd = f"{p.kd_ratio:.2f}"
                    fk = " 🎯" if p.first_kill else ""
                    print(f"        {p.team_name[:3]} | {p.player_name:10} ({p.agent_or_champion:10}): {p.kills:2}/{p.deaths:2}/{p.assists:2} KD:{kd}{fk}")
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE!")
        print("=" * 60)
    
    asyncio.run(main())
