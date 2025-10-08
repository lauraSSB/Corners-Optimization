import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime

class GameStateCalculator:
    """Handles game state calculation for football events - ONLY game state"""

    def calculate_game_state(self, events_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate game state (score difference) for each event
        Game state is calculated from the perspective of the team executing the event
        Positive = winning, Negative = losing, 0 = tied

        Args:
            events_df: DataFrame with event data
            matches_df: DataFrame with match metadata including home/away teams

        Returns:
            DataFrame with added 'game_state' column and match metadata
        """
        if events_df.empty or matches_df.empty:
            return events_df

        # Make copies to avoid modifying original data
        events = events_df.copy()
        matches = matches_df.copy()

        # Get available columns from matches dataframe
        available_columns = ['match_id', 'home_team', 'away_team', 'match_date']

        # First, merge to get available match metadata
        df = events.merge(
            matches[available_columns],
            on='match_id',
            how='left'
        )

        # Proper chronological sorting
        df = self._sort_events_chronologically(df)

        # Calculate game state using the simple and reliable approach
        df = self._calculate_game_state_simple(df)

        return df

    def _sort_events_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort events in proper chronological order within each match"""
        # Create proper timestamp for sorting
        if 'timestamp' in df.columns and df['timestamp'].notna().any():
            # Use timestamp if available (StatsBomb format)
            df['ts'] = pd.to_timedelta(df['timestamp'])
        else:
            # Fall back to minute + second
            df['ts'] = (pd.to_timedelta(df['minute'], unit='m') + 
                       pd.to_timedelta(df.get('second', 0), unit='s'))

        # Sort by match, period, timestamp, and index for tie-breaking
        sort_keys = ['match_id', 'period', 'ts']
        if 'index' in df.columns:
            sort_keys.append('index')

        df = df.sort_values(sort_keys, kind='mergesort').reset_index(drop=True)

        return df

    def _calculate_game_state_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate game state using the simple and reliable approach"""
        # Per event goal deltas from the home perspective

        df['home_goal_evt'] = (
            (df['shot_outcome'] == 'Goal') & (df['team'] == df['home_team']) |
            (df['type'] == 'Own Goal For') & (df['team'] == df['home_team'])
        ).astype(int)

        df['away_goal_evt'] = (
            (df['shot_outcome'] == 'Goal') & (df['team'] == df['away_team']) |
            (df['type'] == 'Own Goal For') & (df['team'] == df['away_team'])
        ).astype(int)

        df['goal_delta'] = df['home_goal_evt'] - df['away_goal_evt']  # +1 if home scores, -1 if away scores, 0 otherwise

        # Game state BEFORE the current event:
        df['game_state'] = df.groupby('match_id')['goal_delta'].cumsum().shift(fill_value=0)

        # Clean up temporary columns
        df = df.drop(columns=['home_goal_evt', 'away_goal_evt', 'goal_delta'], errors='ignore')

        return df