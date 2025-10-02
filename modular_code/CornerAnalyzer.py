import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsbombpy import sb
import warnings
import os
from dotenv import load_dotenv

# Import the updated Corner class
from Corner import Corner

warnings.filterwarnings('ignore')
load_dotenv()

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CornerAnalyzer:
    def __init__(self, username=None, password=None):
        self.username = username or os.getenv('STATSBOMB_USERNAME')
        self.password = password or os.getenv('STATSBOMB_PASSWORD')

        if not self.username or not self.password:
            raise ValueError(
                "Credentials required. Provide them as parameters or set STATSBOMB_USERNAME and STATSBOMB_PASSWORD environment variables")

        print("Credentials configured successfully")

        self.competitions = None
        self.matches = None
        self.corners_data = []

    def get_competitions(self):
        """Get available competitions"""
        self.competitions = sb.competitions(creds={'user': self.username, 'passwd': self.password})
        print("Available competitions: ")
        print(self.competitions[['competition_id', 'competition_name', 'season_name']])
        return self.competitions

    def get_matches(self, competition_id, season_id):
        """Get available matches"""
        self.matches = sb.matches(
            competition_id=competition_id,
            season_id=season_id,
            creds={'user': self.username, 'passwd': self.password}
        )

        print("Available matches: ")
        print(f"\nTotal matches: {len(self.matches)}")
        return self.matches

    def extract_corners_from_season(self, competition_id, season_id, include_frames=True):
        """Extract corner kick data from a season"""
        self.get_matches(competition_id, season_id)
        self.corners_data = []
        counter = 0
        print("\nExtracting corner kick data...")
        
        for idx, match in self.matches.iterrows():
            match_id = match['match_id']
            try:
                if counter > 1:
                    break
                counter += 1
                
                events = sb.events(
                    match_id=match_id,
                    creds={'user': self.username, 'passwd': self.password}
                )

                frames_dict = {}

                if include_frames:
                    try:
                        frames = sb.frames(
                            match_id=match_id,
                            creds={'user': self.username, 'passwd': self.password},
                            fmt='dataframe'
                        )

                        if not frames.empty and 'id' in frames.columns:
                            frames_dict = frames.groupby('id').apply(lambda x: x.to_dict(orient='records')).to_dict()

                    except Exception as e:
                        print(f"No frames available for match {match_id}: {str(e)}")

                corners = events[(events["play_pattern"] == 'From Corner') & (events['pass_type'] == "Corner")]

                for _, event in corners.iterrows():
                    event_id = event.get('id', '')

                    freeze_frame = frames_dict.get(event_id, None)

                    # Convert freeze frame to the expected format for the new Corner class
                    processed_freeze_frame = []
                    if freeze_frame:
                        for frame in freeze_frame:
                            player_data = {
                                'location': frame.get('location', [None, None]),
                                'teammate': frame.get('teammate', False)
                            }
                            processed_freeze_frame.append(player_data)

                    corner = Corner(
                        match_id=match_id,
                        event_id=event_id,
                        team=event.get('team', ''),
                        player=event.get('player', ''),
                        minute=event.get('minute', 0),
                        second=event.get('second', 0),
                        period=event.get('period', 1),
                        location=tuple(event.get('location', [None, None])),
                        end_location=tuple(event.get('pass_end_location', [None, None])) if 'pass_end_location' in event else None,
                        pass_outcome=event.get('pass_outcome', 'Complete'),
                        pass_height=event.get('pass_height', 'Unknown'),
                        pass_type=event.get('pass_type', 'Corner'),
                        body_part=event.get('pass_body_part', 'Unknown'),
                        play_pattern=event.get('play_pattern', 'Unknown'),
                        under_pressure=event.get('under_pressure', False),
                        recipient=event.get('pass_recipient', None),
                        related_events=event.get('related_events', []),
                        freeze_frame=processed_freeze_frame
                    )

                    self.corners_data.append(corner)

                if counter == 1:
                    events_flat = corners.copy()
                    for col in events_flat.columns:
                        if events_flat[col].dtype == 'object':
                            events_flat[col] = events_flat[col].astype(str)

                    events_flat.to_csv('corners.csv', index=False)

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(self.matches)} matches")
                    
            except Exception as e:
                print(f"Error in match: {match_id}: {str(e)}")
                continue

        self.corners_df = pd.DataFrame([corner.to_dict() for corner in self.corners_data])
        print(f"\nTotal corner kicks found: {len(self.corners_df)}")
        return self.corners_df

