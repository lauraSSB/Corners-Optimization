import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import timedelta

class TwentySecondMetrics:
    """Handles calculation of 20-second metrics after each corner event"""
    
    def calculate_20s_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 20-second metrics after each corner event
        
        Args:
            events_df: DataFrame with event data including corner events (from GameStateCalculator)
            This DataFrame should already be chronologically sorted with 'ts' column
        
        Returns:
            DataFrame with added 20-second metric columns for each corner
        """
        if events_df.empty:
            return events_df
        
        df = events_df.copy()
        
        # Check if 'ts' column exists (created by GameStateCalculator)
        if 'ts' not in df.columns:
            # Fallback: create timestamp from minute and second if needed
            df['ts'] = (pd.to_timedelta(df['minute'], unit='m') + 
                        pd.to_timedelta(df.get('second', 0), unit='s'))
        
        # Identify corner events
        corner_events = self._identify_corner_events(df)
        
        if corner_events.empty:
            # Add empty columns for consistency
            df['xg_20s'] = 0.0
            df['xg_20s_def'] = 0.0
            df['goal_20s'] = 0
            df['goal_20s_def'] = 0
            return df
        
        # Calculate 20-second metrics for each corner
        corner_metrics = []
        
        for idx, corner_row in corner_events.iterrows():
            metrics = self._calculate_metrics_for_corner(df, corner_row, idx)
            corner_metrics.append(metrics)
        
        # Create metrics DataFrame and merge back
        metrics_df = pd.DataFrame(corner_metrics)
        
        # Merge metrics back to original DataFrame
        result_df = df.merge(
            metrics_df, 
            left_index=True, 
            right_on='original_index', 
            how='left', 
            suffixes=('', '_20s')
        )
        
        # Fill NaN values for non-corner events
        result_df['xg_20s'] = result_df['xg_20s'].fillna(0.0)
        result_df['xg_20s_def'] = result_df['xg_20s_def'].fillna(0.0)
        result_df['goal_20s'] = result_df['goal_20s'].fillna(0)
        result_df['goal_20s_def'] = result_df['goal_20s_def'].fillna(0)
        
        # Drop the temporary index column
        result_df = result_df.drop(columns=['original_index'], errors='ignore')
        
        return result_df
    
    def _identify_corner_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify corner pass execution events"""
        mask_corner = (
            (df.get('play_pattern') == "From Corner") &
            (df.get('pass_type') == "Corner") &
            (df.get('type') == "Pass")
        )
        
        corner_events = df[mask_corner].copy()
        return corner_events
    
    def _calculate_metrics_for_corner(self, df: pd.DataFrame, corner_row: pd.Series, corner_idx: int) -> Dict[str, Any]:
        """Calculate 20-second metrics for a specific corner event"""
        metrics = {
            'original_index': corner_idx,
            'xg_20s': 0.0,    # xG for attacking team in 20s window
            'xg_20s_def': 0.0,    # xG for defending team in 20s window
            'goal_20s': 0,    # Goal scored by attacking team in 20s window
            'goal_20s_def': 0    # Goal scored by defending team in 20s window
        }
        
        # Get corner timestamp and team information
        corner_ts = corner_row['ts']
        match_id = corner_row['match_id']
        period = corner_row['period']
        corner_team = corner_row['team']
        
        if pd.isna(corner_ts):
            return metrics
        
        # Find events within 20 seconds after this corner
        mask_20s = (
            (df['match_id'] == match_id) &
            (df['period'] == period) &
            (df['ts'] > corner_ts) &
            (df['ts'] <= corner_ts + timedelta(seconds=20))
        )
        
        events_20s = df[mask_20s]
        
        if events_20s.empty:
            return metrics
        
        # Calculate xG metrics - using shot_statsbomb_xg column from sample data
        if 'shot_statsbomb_xg' in events_20s.columns:
            # xG for attacking team (corner team)
            att_xg_mask = events_20s['team'] == corner_team
            metrics['xg_20s'] = events_20s.loc[att_xg_mask, 'shot_statsbomb_xg'].sum()
            
            # xG for defending team (opponent)
            def_xg_mask = events_20s['team'] != corner_team
            metrics['xg_20s_def'] = events_20s.loc[def_xg_mask, 'shot_statsbomb_xg'].sum()
            
            # Check for goals - using shot_outcome column from sample data
            goal_events = events_20s[
                (events_20s['shot_outcome'] == 'Goal') |
                (events_20s['type'] == 'Own Goal For')
            ]
            
            if not goal_events.empty:
                # Goals by attacking team
                att_goals = goal_events[goal_events['team'] == corner_team]
                if not att_goals.empty:
                    metrics['goal_20s'] = 1
                
                # Goals by defending team
                def_goals = goal_events[goal_events['team'] != corner_team]
                if not def_goals.empty:
                    metrics['goal_20s_def'] = 1
        
        return metrics