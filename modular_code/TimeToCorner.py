import pandas as pd
import numpy as np
from typing import Optional

class TimeToCorner:
    """Handles calculation of corner execution time and categorization using pre-processed DataFrame"""
    
    def calculate_corner_execution_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate corner execution time and create categorical labels
        Assumes DataFrame is already processed by GameStateCalculator (sorted chronologically)
        
        Args:
            df: DataFrame with event data including timestamp, play_pattern, pass_type, type
                 This should be the output from GameStateCalculator.calculate_game_state()
            
        Returns:
            DataFrame with added 'corner_execution_time_raw' and 'corner_execution_time_label' columns
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Parse timestamp to seconds
        df['time_seconds'] = df['timestamp'].apply(self._ts_to_seconds)
        
        # Since GameStateCalculator already sorted chronologically, we don't need to sort again
        # The DataFrame should already be properly sorted by match_id, period, and timestamp
        
        # Identify corner-pass execution rows
        mask_corner_exec = (
            (df.get('play_pattern') == "From Corner") &
            (df.get('pass_type') == "Corner") &
            (df.get('type') == "Pass")
        )
        
        # Previous event time within the same match and period
        # Use the existing chronological order from GameStateCalculator
        df['prev_time_in_match'] = df.groupby(['match_id', 'period'])['time_seconds'].shift(1)
        
        # Calculate raw corner execution time
        df['corner_execution_time_raw'] = None
        df.loc[mask_corner_exec, 'corner_execution_time_raw'] = (
            df.loc[mask_corner_exec, 'time_seconds'] - df.loc[mask_corner_exec, 'prev_time_in_match']
        )
        
        # Create categorical labels
        df['corner_execution_time_label'] = df['corner_execution_time_raw'].apply(self._categorize_execution_time)
        
        # Drop helper columns
        drop_cols = [c for c in ['time_seconds', 'prev_time_in_match'] if c in df.columns]
        df = df.drop(columns=drop_cols)
        
        return df
    
    def _ts_to_seconds(self, ts: Optional[str]) -> Optional[float]:
        """Parse timestamp "HH:MM:SS.mmm" -> total seconds (float)"""
        if pd.isna(ts):
            return None
        try:
            h, m, s = ts.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, AttributeError):
            return None
    
    def _categorize_execution_time(self, time_seconds: Optional[float]) -> Optional[str]:
        """Categorize execution time into buckets"""
        if pd.isna(time_seconds) or time_seconds is None:
            return None
        
        if time_seconds <= 5:
            return "0-5 seconds"
        elif time_seconds <= 10:
            return "5-10 seconds"
        elif time_seconds <= 15:
            return "10-15 seconds"
        elif time_seconds <= 20:
            return "15-20 seconds"
        elif time_seconds <= 30:
            return "20-30 seconds"
        elif time_seconds <= 40:
            return "30-40 seconds"
        else:
            return "+40 seconds"