import pandas as pd
import warnings
import os
from datetime import datetime
from dotenv import load_dotenv

from DataDownloader import DataDownloader
from GameStateCalculator import GameStateCalculator
from TwentySecondMetrics import TwentySecondMetrics
from TimeToCorner import TimeToCorner
from CornerAnalyzer import CornerAnalyzer

warnings.filterwarnings('ignore')
load_dotenv()

def main():
    """Main function to download Liga MX data, calculate game state, corner execution time, and corner analysis"""

    try:
        downloader = DataDownloader()
        game_state_calculator = GameStateCalculator()
        twenty_second_metrics = TwentySecondMetrics()
        time_to_corner = TimeToCorner()
        corner_analyzer = CornerAnalyzer(downloader)

        comps = downloader.get_competitions()
        comp_id = comps.iloc[0]['competition_id']
        season_id = comps.iloc[0]['season_id']

        matches_df = downloader.get_matches(comp_id, season_id)
        all_corner_analysis = []

        max_matches = min(5, len(matches_df))

        print(f"Processing {max_matches} matches...")

        for i, match_row in matches_df.head(max_matches).iterrows():
            match_id = match_row['match_id']
            match_name = match_row.get('match_name', f'Match {match_id}')
            print(f"Processing: {match_name}")

            events_df = downloader.get_match_events(match_id)

            # Calculate game state
            events_with_state = game_state_calculator.calculate_game_state(events_df, matches_df)

            # Calculate 20-second metrics after corners
            events_with_20s_metrics = twenty_second_metrics.calculate_20s_metrics(events_with_state)

            # Calculate corner execution time
            events_with_corner_time = time_to_corner.calculate_corner_execution_time(events_with_20s_metrics)

            # Perform corner analysis on this match's data
            corner_analysis_df = corner_analyzer.analyze_corners_from_processed_data(events_with_corner_time)
          
            if not corner_analysis_df.empty:
                all_corner_analysis.append(corner_analysis_df)
                print(f"Found {len(corner_analysis_df)} corner events")
            else:
                print(f"No corner events found in this match")

        if all_corner_analysis:
            combined_corners = pd.concat(all_corner_analysis, ignore_index=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the unified final DataFrame with only corner events and zone data
            unified_filename = f"unified_corner_analysis_{timestamp}.csv"
            combined_corners.to_csv(unified_filename, index=False)

        else:
            print("No corner events were found in any match")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()