import pandas as pd
from statsbombpy import sb
import os
import time
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
        """Get all events for a specific match with retry logic"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching events for match {match_id} (attempt {attempt + 1}/{max_retries})...")
                events = sb.events(
                    match_id=match_id,
                    creds={'user': self.username, 'passwd': self.password}
                )
                
                # Check if we got valid data
                if events is not None and not events.empty:
                    events['match_id'] = match_id
                    print(f"✓ Successfully fetched {len(events)} events for match {match_id}")
                    return events
                else:
                    print(f"⚠ No events data returned for match {match_id}")
                    return pd.DataFrame()
                    
            except Exception as e:
                print(f"✗ Error fetching events for match {match_id} (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to fetch events for match {match_id} after {max_retries} attempts")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def get_match_frames(self, match_id: int) -> Dict:
        """Get freeze frames for a specific match with retry logic"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching frames for match {match_id} (attempt {attempt + 1}/{max_retries})...")
                frames = sb.frames(
                    match_id=match_id,
                    creds={'user': self.username, 'passwd': self.password},
                    fmt='dataframe'
                )

                if not frames.empty and 'id' in frames.columns:
                    frames_dict = frames.groupby('id').apply(
                        lambda x: x[['location', 'teammate', 'keeper']].to_dict('records')
                    ).to_dict()
                    print(f"✓ Successfully fetched frames for match {match_id}")
                    return frames_dict
                else:
                    print(f"⚠ No frames data returned for match {match_id}")
                    return {}
                    
            except Exception as e:
                print(f"✗ Error fetching frames for match {match_id} (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to fetch frames for match {match_id} after {max_retries} attempts")
                    return {}
        
        return {}