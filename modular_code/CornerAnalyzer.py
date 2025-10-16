import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from Corner import Corner
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')
load_dotenv()

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CornerAnalyzer:
    """Handles corner kick analysis - integrates with existing pipeline"""

    def __init__(self, downloader: DataDownloader = None):
        self.downloader = downloader or DataDownloader()
        self.corners_data: List[Corner] = []
        self.corners_df: Optional[pd.DataFrame] = None

    def _get_p1_events(self, corner_event, processed_events_df):
        """
        Identify P1 events - events at the IMMEDIATE next timestamp after P0
        
        STRICT RULE: P1 = All events at the first timestamp after P0, NO FURTHER
        - Get P0 timestamp
        - Find the immediate next timestamp (P1 timestamp)
        - Check ONLY events at P1 timestamp for freeze frames
        - If no freeze frame at P1 timestamp, return empty list
        - NEVER check events beyond P1 timestamp
        
        Args:
            corner_event: The corner kick event (P0)
            processed_events_df: DataFrame with all processed events

        Returns:
            List with one P1 event (with freeze frame) or empty list
        """
        p1_events = []

        # Get P0 (corner) timestamp and match_id
        p0_timestamp = corner_event.get('timestamp', 0)
        match_id = corner_event.get('match_id')
        corner_event_id = corner_event.get('id', '')

        print(f"\n[DEBUG] Processing corner {corner_event_id}")
        print(f"[DEBUG] P0 timestamp: {p0_timestamp}")

        # Find ALL events after P0, sorted by timestamp
        subsequent_events = processed_events_df[
            (processed_events_df['match_id'] == match_id) &
            (processed_events_df['timestamp'] > p0_timestamp) &
            (processed_events_df['id'] != corner_event_id)
        ].sort_values(['timestamp', 'index'])

        if subsequent_events.empty:
            print(f"[DEBUG] No events after P0")
            return p1_events

        # Get the IMMEDIATE next timestamp - this is P1 timestamp
        p1_timestamp = subsequent_events.iloc[0]['timestamp']
        print(f"[DEBUG] P1 timestamp: {p1_timestamp}")

        # STRICT CONSTRAINT: Get ONLY events at P1 timestamp (not beyond)
        p1_candidates = subsequent_events[
            subsequent_events['timestamp'] == p1_timestamp
        ].sort_values('index')

        print(f"[DEBUG] Found {len(p1_candidates)} events at P1 timestamp")

        # Check each event at P1 timestamp for freeze frame
        for idx, (_, p1_row) in enumerate(p1_candidates.iterrows()):
            p1_event_id = p1_row.get('id', '')
            p1_event_type = p1_row.get('type', '')
            
            print(f"[DEBUG] Checking P1 event {idx+1}/{len(p1_candidates)}: {p1_event_id} (type: {p1_event_type})")

            try:
                # Get freeze frame data for this specific P1 event
                frames_dict = self.downloader.get_match_frames(match_id)
                p1_freeze_frame = frames_dict.get(p1_event_id, [])

                if p1_freeze_frame:
                    print(f"[DEBUG] ✓ Found freeze frame at P1 event {p1_event_id} with {len(p1_freeze_frame)} players")
                    
                    p1_data = {
                        'p1_event_id': p1_event_id,
                        'p1_timestamp': p1_row.get('timestamp', 0),
                        'p1_type': p1_event_type,
                        'p1_player': p1_row.get('player', ''),
                        'p1_team': p1_row.get('team', ''),
                        'p1_location': p1_row.get('location', [None, None]),
                        'p1_freeze_frame': p1_freeze_frame,
                        'p1_related_events': p1_row.get('related_events', []),
                        'p1_num_defenders_on_goal_side': p1_row.get('num_defenders_on_goal_side_of_actor', 0),
                        'p1_visible_opponents': p1_row.get('visible_opponents', 0),
                        'p1_visible_teammates': p1_row.get('visible_teammates', 0)
                    }

                    # Add goalkeeper positioning data
                    goalkeeper_data = self._extract_goalkeeper_positioning(p1_freeze_frame)
                    p1_data.update(goalkeeper_data)

                    p1_events.append(p1_data)
                    
                    # Found freeze frame at P1, stop searching
                    print(f"[DEBUG] Stopping search - freeze frame found at P1")
                    break
                else:
                    print(f"[DEBUG] ✗ No freeze frame for event {p1_event_id}")

            except Exception as e:
                print(f"[DEBUG] Error processing P1 event {p1_event_id}: {str(e)}")
                continue

        # If we exit the loop without finding a freeze frame at P1
        if not p1_events:
            print(f"[DEBUG] No freeze frame found at P1 timestamp - P1 data will be empty")

        return p1_events

    def _extract_goalkeeper_positioning(self, freeze_frame):
        """Extract goalkeeper positioning data from freeze frame"""
        goalkeeper_data = {
            'goalkeeper_x': None,
            'goalkeeper_y': None,
            'goalkeeper_distance_to_goal': None,
            'goalkeeper_angle_to_goal': None
        }

        for player in freeze_frame:
            if player.get('position', {}).get('name') == 'Goalkeeper':
                location = player.get('location', [None, None])
                if location and len(location) >= 2:
                    goalkeeper_data['goalkeeper_x'] = location[0]
                    goalkeeper_data['goalkeeper_y'] = location[1]

                    # Calculate distance to goal (assuming goal at [120, 40])
                    if location[0] is not None and location[1] is not None:
                        goal_x, goal_y = 120, 40  # Typical goal position
                        distance = ((location[0] - goal_x) ** 2 + (location[1] - goal_y) ** 2) ** 0.5
                        goalkeeper_data['goalkeeper_distance_to_goal'] = distance

                        # Calculate angle to goal
                        dx = goal_x - location[0]
                        dy = goal_y - location[1]
                        angle = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0
                        goalkeeper_data['goalkeeper_angle_to_goal'] = angle
                break

        return goalkeeper_data

    def analyze_corners_from_processed_data(self, processed_events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze corners from already processed data (with game state and corner execution time)

        Args:
            processed_events_df: DataFrame with events that have been processed by
                                GameStateCalculator and TimeToCorner

        Returns:
            DataFrame with detailed corner analysis including P0 and P1 data
        """
        if processed_events_df.empty:
            print("No processed events to analyze")
            return pd.DataFrame()

        print(f"Analyzing corners from {len(processed_events_df)} processed events...")

        # Filter corner events from the processed data
        corner_events = processed_events_df[
            (processed_events_df["play_pattern"] == 'From Corner') &
            (processed_events_df['type'] == 'Pass') &
            (processed_events_df['pass_type'] == "Corner")
        ].copy()

        print(f"Found {len(corner_events)} corner kick events")

        if corner_events.empty:
            print("No corner kick events found in the processed data")
            return pd.DataFrame()

        self.corners_data = []

        # Process each corner event to extract detailed analysis
        for _, event in corner_events.iterrows():
            match_id = event.get('match_id')
            event_id = event.get('id', '')

            try:
                # Get freeze frame data for this corner
                frames_dict = self.downloader.get_match_frames(match_id)
                freeze_frame = frames_dict.get(event_id, [])

                # Get P1 events for this corner (with strict P1 constraint)
                p1_events = self._get_p1_events(event, processed_events_df)

                # Prepare location data
                location = event.get('location', [None, None])
                end_location = event.get('pass_end_location', [None, None])

                # Create Corner object with detailed analysis
                corner = Corner(
                    match_id=match_id,
                    event_id=event_id,
                    team=event.get('team', ''),
                    player=event.get('player', ''),
                    minute=event.get('minute', 0),
                    second=event.get('second', 0),
                    period=event.get('period', 1),
                    location=tuple(location) if location and len(location) >= 2 else None,
                    end_location=tuple(end_location) if end_location and len(end_location) >= 2 else None,
                    pass_outcome=event.get('pass_outcome', 'Complete'),
                    pass_height=event.get('pass_height', 'Unknown'),
                    pass_length=event.get('pass_length', 'Unknown'),
                    pass_technique = event.get('pass_technique', 'Unknown'),
                    pass_type=event.get('pass_type', 'Corner'),
                    body_part=event.get('pass_body_part', 'Unknown'),
                    play_pattern=event.get('play_pattern', 'Unknown'),
                    recipient=event.get('pass_recipient', None),
                    related_events=event.get('related_events', []),
                    freeze_frame=freeze_frame
                )
                
                # Add P1 events to the corner object
                corner.p1_events = p1_events

                self.corners_data.append(corner)

            except Exception as e:
                print(f"Error processing corner event {event_id}: {str(e)}")
                continue

        # Create final DataFrame with all corner analysis
        if self.corners_data:
            # Convert corners to dictionary format including P1 data
            corner_dicts = []
            for corner in self.corners_data:
                corner_dict = corner.to_dict()

                # Add P1 data to the corner dictionary
                if corner.p1_events:
                    # For now, take the first P1 event (can be extended to handle multiple)
                    p1_data = corner.p1_events[0]

                    # Add P1 columns with prefix
                    for key, value in p1_data.items():
                        corner_dict[f'p1_{key}'] = value

                corner_dicts.append(corner_dict)

            self.corners_df = pd.DataFrame(corner_dicts)

            # Get only the specific columns you requested from processed data
            corner_ids = [corner.event_id for corner in self.corners_data]

            # Extract only the specific columns you want to incorporate
            additional_columns = processed_events_df.loc[
                processed_events_df['id'].isin(corner_ids),
                ['id', 'game_state', 'corner_execution_time_raw', 'corner_execution_time_label',
                 'match_date', 'home_team', 'away_team', 'xg_20s', 'xg_20s_def', 'goal_20s', 'goal_20s_def']
            ]

            # Simple merge without suffixes - just add the specific columns
            self.corners_df = self.corners_df.merge(
                additional_columns,
                left_on='event_id',
                right_on='id',
                how='left'
            )

            # Remove the duplicate 'id' column
            if 'id' in self.corners_df.columns:
                self.corners_df = self.corners_df.drop(columns=['id'])

            print(f"Successfully analyzed {len(self.corners_df)} corner kicks with P1 data")

            return self.corners_df
        else:
            print("No corner data was successfully processed")
            return pd.DataFrame()

    def save_to_csv(self, filename: str = 'corners_analysis.csv') -> Optional[str]:
        """Save corner data to CSV file"""
        if self.corners_df is not None and not self.corners_df.empty:
            self.corners_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Data saved to: {filename}")
            return filename
        else:
            print("No data to save")
            return None

# Example usage
if __name__ == "__main__":
    # This demonstrates how to use the analyzer with existing processed data
    print("CornerAnalyzer is designed to work with data processed by the main pipeline")
    print("Use analyze_corners_from_processed_data() method with data from TimeToCorner")
    print("Example: analyzer.analyze_corners_from_processed_data(processed_events_df)")