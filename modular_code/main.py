import pandas as pd
import warnings
import os
import pickle
import hashlib
from datetime import datetime
from dotenv import load_dotenv

from DataDownloader import DataDownloader
from GameStateCalculator import GameStateCalculator
from TwentySecondMetrics import TwentySecondMetrics
from TimeToCorner import TimeToCorner
from CornerAnalyzer import CornerAnalyzer
from CornerP1Analyzer import CornerP1Analyzer
from CornerNextBallLocations import CornerNextBallLocations


warnings.filterwarnings('ignore')
load_dotenv()

class CachedDataDownloader(DataDownloader):
    def __init__(self, cache_dir="./statsbomb_cache", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, endpoint, **params):
        key_str = f"{endpoint}_{str(params)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_match_events(self, match_id: int) -> pd.DataFrame:
        cache_key = self._get_cache_key("events", match_id=match_id)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Try cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                print(f"✓ Loaded events for match {match_id} from cache")
                return pickle.load(f)
        
        # Fetch from API
        events = super().get_match_events(match_id)
        
        # Cache the result
        if not events.empty:
            with open(cache_file, 'wb') as f:
                pickle.dump(events, f)
        
        return events
    
    def get_match_frames(self, match_id: int) -> dict:
        cache_key = self._get_cache_key("frames", match_id=match_id)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                print(f"✓ Loaded frames for match {match_id} from cache")
                return pickle.load(f)
        
        frames = super().get_match_frames(match_id)
        
        if frames:
            with open(cache_file, 'wb') as f:
                pickle.dump(frames, f)
        
        return frames

def main():
    """Main function to download Liga MX data, calculate game state, corner execution time, and corner analysis"""

    try:
        downloader = CachedDataDownloader(cache_dir="./statsbomb_cache")
        game_state_calculator = GameStateCalculator()
        twenty_second_metrics = TwentySecondMetrics()
        time_to_corner = TimeToCorner()
        corner_analyzer = CornerAnalyzer(downloader)
        p1_analyzer = CornerP1Analyzer(downloader)
        next_ball_locations = CornerNextBallLocations(downloader)


        comps = downloader.get_competitions()
        comp_id = comps.iloc[0]['competition_id']
        season_id = comps.iloc[0]['season_id']

        matches_df = downloader.get_matches(comp_id, season_id)
        all_corner_analysis = []

        max_matches = min(1, len(matches_df))

        print(f"Processing {max_matches} matches...")

        processed_matches = 0
        successful_matches = 0
        failed_matches = 0

        for i, match_row in matches_df.head(max_matches).iterrows():
            match_id = match_row['match_id']
            match_name = match_row.get('match_name', f'Match {match_id}')
            processed_matches += 1
            print(f"Processing: {match_name} ({processed_matches}/{max_matches})")

            events_df = downloader.get_match_events(match_id)

            # Skip if no events data (API failure)
            if events_df.empty:
                print(f"⚠ No events data for match {match_id}, skipping...")
                failed_matches += 1
                continue

            try:
                # Calculate game state
                events_with_state = game_state_calculator.calculate_game_state(events_df, matches_df)

                # Calculate 20-second metrics after corners
                events_with_20s_metrics = twenty_second_metrics.calculate_20s_metrics(events_with_state)

                # Calculate corner execution time
                events_with_corner_time = time_to_corner.calculate_corner_execution_time(events_with_20s_metrics)

                # Perform P0 corner analysis on this match's data
                p0_df = corner_analyzer.analyze_corners_from_processed_data(events_with_corner_time)
                
                if not p0_df.empty:
                    # Perform P1 analysis for this match
                    p1_df = p1_analyzer.analyze_p1_events(events_with_corner_time, corner_analyzer.corners_data)
                    
                    # Merge P0 and P1 data
                    combined_df = p1_analyzer.merge_p0_p1_data(p0_df, p1_df)

                    # Perform next five ball receipts actions analysis for this match
                    next_ball_locations_df = next_ball_locations.find_next_ball_receipts(combined_df, events_df)
                    final_df = pd.merge(combined_df, next_ball_locations_df, on="event_id", how="left")

                    # Merge P0 and P1 data
                    all_corner_analysis.append(final_df)
                    
                    print(f"Found {len(p0_df)} corner events, {len(p1_df) if p1_df is not None else 0} P1 events")
                    successful_matches += 1
                else:
                    print(f"No corner events found in this match")
                    successful_matches += 1  # Still counts as successful if we got events but no corners

            except Exception as e:
                print(f"❌ Error processing events for match {match_id}: {str(e)}")
                failed_matches += 1
                continue

        # Final summary
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total matches processed: {processed_matches}")
        print(f"Successful matches: {successful_matches}")
        print(f"Failed matches: {failed_matches}")
        print(f"Success rate: {(successful_matches/processed_matches)*100:.1f}%")

        if all_corner_analysis:
            combined_corners = pd.concat(all_corner_analysis, ignore_index=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the unified final DataFrame with P0 and P1 data
            unified_filename = f"unified_corner_analysis_p0_p1_{timestamp}.csv"
            combined_corners.to_csv(unified_filename, index=False)
            print(f"Saved combined P0+P1 analysis to: {unified_filename}")
            print(f"Total corners analyzed: {len(combined_corners)}")

        else:
            print("No corner events were found in any match")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()