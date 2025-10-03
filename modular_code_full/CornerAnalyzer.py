import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsbombpy import sb
import warnings
import os
from dotenv import load_dotenv

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
                "Credentials required. Provide them as parameters or in environment variables STATSBOMB_USERNAME and STATSBOMB_PASSWORD")

        print("Credentials configured successfully")

        self.competitions = None
        self.matches = None
        self.corners_data = []
        self.corners_df = None

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

    def extract_corners_from_season(self, competition_id, season_id, include_frames=True, max_matches=None):
        """
        Extract corner kick data from a complete season
        
        Args:
            competition_id: Competition ID
            season_id: Season ID
            include_frames: Include freeze frame data
            max_matches: Maximum number of matches to process (None for all)
        """
        self.get_matches(competition_id, season_id)
        self.corners_data = []
        
        if max_matches:
            matches_to_process = self.matches.head(max_matches)
        else:
            matches_to_process = self.matches
        
        print(f"\nExtracting corner kick data from {len(matches_to_process)} matches...")
        
        for idx, match in matches_to_process.iterrows():
            match_id = match['match_id']
            match_name = match.get('match_name', f'Match {match_id}')
            
            try:
                print(f"\nProcessing match: {match_name} (ID: {match_id})")
                
                # Get match events
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
                            # Create frames dictionary by event_id
                            frames_dict = frames.groupby('id').apply(
                                lambda x: x[['location', 'teammate']].to_dict('records')
                            ).to_dict()
                            print(f"  - {len(frames_dict)} freeze frames available")

                    except Exception as e:
                        print(f"  - No frames available for match {match_id}: {str(e)}")

                # Filter corner kicks
                corners = events[(events["play_pattern"] == 'From Corner') & 
                                 (events['type'] == 'Pass') & 
                                 (events['pass_type'] == "Corner")]

                print(f"  - {len(corners)} corner kicks found")

                for _, event in corners.iterrows():
                    event_id = event.get('id', '')
                    
                    # Get freeze frame for this specific event
                    freeze_frame = frames_dict.get(event_id, [])
                    
                    # Prepare location data
                    location = event.get('location', [None, None])
                    end_location = event.get('pass_end_location', [None, None])
                    
                    # Create Corner object
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
                        pass_type=event.get('pass_type', 'Corner'),
                        body_part=event.get('pass_body_part', 'Unknown'),
                        play_pattern=event.get('play_pattern', 'Unknown'),
                        under_pressure=event.get('under_pressure', False),
                        recipient=event.get('pass_recipient', None),
                        related_events=event.get('related_events', []),
                        freeze_frame=freeze_frame
                    )

                    self.corners_data.append(corner)

                # Progress tracking
                if (idx + 1) % 5 == 0:
                    print(f"Processed {idx + 1}/{len(matches_to_process)} matches")
                    
            except Exception as e:
                print(f"Error in match {match_id}: {str(e)}")
                continue

        # Create final DataFrame
        self.corners_df = pd.DataFrame([corner.to_dict() for corner in self.corners_data])
        print(f"\n✅ Extraction completed!")
        print(f"Total corner kicks processed: {len(self.corners_df)}")
        print(f"Total matches processed: {len(matches_to_process)}")
        
        return self.corners_df

    def save_to_csv(self, filename='corners_analysis.csv'):
        """Save corner data to CSV file"""
        if self.corners_df is not None:
            self.corners_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Data saved to: {filename}")
            return filename
        else:
            print("No data to save")
            return None

    def get_summary_stats(self):
        """Get summary statistics of corners"""
        if self.corners_df is None or self.corners_df.empty:
            print("No corner data available")
            return None
        
        summary = {
            'total_corners': len(self.corners_df),
            'unique_matches': self.corners_df['match_id'].nunique(),
            'unique_teams': self.corners_df['team'].nunique(),
            'unique_players': self.corners_df['player'].nunique(),
            'corners_by_side': self.corners_df['corner_side'].value_counts().to_dict(),
            'successful_corners': len(self.corners_df[self.corners_df['pass_outcome'] == 'Complete']),
            'success_rate': (len(self.corners_df[self.corners_df['pass_outcome'] == 'Complete']) / len(self.corners_df)) * 100
        }
        
        print("\n📊 Summary Statistics:")
        print("=" * 50)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key.replace('_', ' ').title()}: {value}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        return summary

    def analyze_zone_distribution(self):
        """Analyze player distribution in zones"""
        if self.corners_df is None or self.corners_df.empty:
            print("No corner data available")
            return None
        
        # Calculate averages by zone
        zone_columns = [col for col in self.corners_df.columns if col.startswith('zone_') and col.endswith('_total')]
        
        zone_stats = {}
        for zone_col in zone_columns:
            zone_id = zone_col.split('_')[1]
            zone_name_col = f'zone_{zone_id}_name'
            
            if zone_name_col in self.corners_df.columns:
                zone_name = self.corners_df[zone_name_col].iloc[0]
                avg_players = self.corners_df[zone_col].mean()
                zone_stats[zone_id] = {
                    'name': zone_name,
                    'avg_players': round(avg_players, 2),
                    'total_occurrences': self.corners_df[self.corners_df[zone_col] > 0].shape[0]
                }
        
        print("\n🎯 Average Zone Distribution:")
        print("=" * 60)
        for zone_id, stats in sorted(zone_stats.items(), key=lambda x: x[1]['avg_players'], reverse=True):
            print(f"Zone {zone_id} ({stats['name']}): {stats['avg_players']} players on average")
        
        return zone_stats

    def plot_corner_distribution(self):
        """Create corner distribution chart by side"""
        if self.corners_df is None or self.corners_df.empty:
            print("No corner data available")
            return None
        
        plt.figure(figsize=(10, 6))
        corner_counts = self.corners_df['corner_side'].value_counts()
        
        plt.bar(corner_counts.index, corner_counts.values, color=['skyblue', 'lightcoral'])
        plt.title('Corner Kick Distribution by Side')
        plt.xlabel('Corner Side')
        plt.ylabel('Number of Corners')
        plt.grid(axis='y', alpha=0.3)
        
        for i, count in enumerate(corner_counts.values):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CornerAnalyzer()
    
    try:
        # Get competitions
        comps = analyzer.get_competitions()
        
        # Select first competition and season (modify as needed)
        comp_id = comps.iloc[0]['competition_id']
        season_id = comps.iloc[0]['season_id']
        
        # Extract corners (process only 2 matches for testing)
        corners_df = analyzer.extract_corners_from_season(comp_id, season_id, max_matches=2)
        
        # Show statistics
        analyzer.get_summary_stats()
        analyzer.analyze_zone_distribution()
        
        # Save data
        analyzer.save_to_csv()
        
        # Show chart
        analyzer.plot_corner_distribution()
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

