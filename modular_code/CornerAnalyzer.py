import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

    def analyze_corners_from_processed_data(self, processed_events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze corners from already processed data (with game state and corner execution time)
        
        Args:
            processed_events_df: DataFrame with events that have been processed by 
                               GameStateCalculator and TimeToCorner
        
        Returns:
            DataFrame with detailed corner analysis including zone data
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
                
                self.corners_data.append(corner)
                
            except Exception as e:
                print(f"Error processing corner event {event_id}: {str(e)}")
                continue
        
        # Create final DataFrame with all corner analysis
        if self.corners_data:
            self.corners_df = pd.DataFrame([corner.to_dict() for corner in self.corners_data])
            
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
            
            print(f"Successfully analyzed {len(self.corners_df)} corner kicks")
            
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