import pandas as pd
from statsbombpy import sb
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class DataDownloader:
    """Handles downloading data from StatsBomb API - ONLY data downloading"""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv('STATSBOMB_USERNAME')
        self.password = password or os.getenv('STATSBOMB_PASSWORD')

        if not self.username or not self.password:
            raise ValueError(
                "Credentials required. Provide them as parameters or in environment variables STATSBOMB_USERNAME and STATSBOMB_PASSWORD"
            )

    def get_competitions(self) -> pd.DataFrame:
        """Get available competitions"""
        return sb.competitions(creds={'user': self.username, 'passwd': self.password})

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Get matches for a specific competition and season"""
        return sb.matches(
            competition_id=competition_id,
            season_id=season_id,
            creds={'user': self.username, 'passwd': self.password}
        )

    def get_match_events(self, match_id: int) -> pd.DataFrame:
        """Get all events for a specific match"""
        events = sb.events(
            match_id=match_id,
            creds={'user': self.username, 'passwd': self.password}
        )
        events['match_id'] = match_id
        return events

    def get_match_frames(self, match_id: int) -> Dict:
        """Get freeze frames for a specific match"""
        try:
            frames = sb.frames(
                match_id=match_id,
                creds={'user': self.username, 'passwd': self.password},
                fmt='dataframe'
            )

            if not frames.empty and 'id' in frames.columns:
                return frames.groupby('id').apply(
                    lambda x: x[['location', 'teammate']].to_dict('records')
                ).to_dict()
            return {}
        except Exception:
            return {}

    def get_next_related_event(self, match_id: int, event_id: str) -> Dict:
        """Get next related event for a specific event_id"""
        try:
            events = sb.events(
                match_id=match_id,
                creds={'user': self.username, 'passwd': self.password},
                fmt='dataframe'
            )

            mask = events["related_events"].apply(lambda v: isinstance(v, list) and event_id in v)
            related_events = events.loc[mask].copy()

            if related_events.empty:
                return {}

            related_events = related_events.sort_values(by="timestamp").reset_index(drop=True)
            return related_events.iloc[0].to_dict()

        except Exception as e:
            print(type(e).__name__, ":", e)
            return {}
