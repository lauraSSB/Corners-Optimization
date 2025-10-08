import pandas as pd
import warnings
import os
from datetime import datetime
from dotenv import load_dotenv

from DataDownloader import DataDownloader
from GameStateCalculator import GameStateCalculator
from TimeToCorner import TimeToCorner
from CornerAnalyzer import CornerAnalyzer

warnings.filterwarnings('ignore')
load_dotenv()

def main():
    """Main function to download Liga MX data, calculate game state, corner execution time, and corner analysis"""
    
    try:
        downloader = DataDownloader()
        game_state_calculator = GameStateCalculator()
        time_to_corner = TimeToCorner()
        corner_analyzer = CornerAnalyzer(downloader)
        
        comps = downloader.get_competitions()
        comp_id = comps.iloc[0]['competition_id']
        season_id = comps.iloc[0]['season_id']
        
        matches_df = downloader.get_matches(comp_id, season_id)
        all_events_with_corner_time = []
        
        max_matches = min(5, len(matches_df))
        
        print(f"Processing {max_matches} matches...")
        
        for i, match_row in matches_df.head(max_matches).iterrows():
            match_id = match_row['match_id']
            match_name = match_row.get('match_name', f'Match {match_id}')
            print(f"Processing: {match_name}")
            
            events_df = downloader.get_match_events(match_id)
            
            # Calculate game state
            events_with_state = game_state_calculator.calculate_game_state(events_df, matches_df)
            
            # Calculate corner execution time
            events_with_corner_time = time_to_corner.calculate_corner_execution_time(events_with_state)
            
            all_events_with_corner_time.append(events_with_corner_time)
            
        if all_events_with_corner_time:
            combined_events = pd.concat(all_events_with_corner_time, ignore_index=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Perform corner analysis on the processed data
            print("\n" + "="*60)
            print("PERFORMING CORNER ANALYSIS")
            print("="*60)
            
            corner_analysis_df = corner_analyzer.analyze_corners_from_processed_data(combined_events)
            
            if not corner_analysis_df.empty:
                # The CornerAnalyzer already incorporates game_state and execution time columns
                # No need for additional merging - this is our final unified DataFrame
                unified_df = corner_analysis_df
                
                # Save the unified final DataFrame
                unified_filename = f"unified_corner_analysis_{timestamp}.csv"
                unified_df.to_csv(unified_filename, index=False)
                
                print(f"\nUnified analysis saved to: {unified_filename}")
                print(f"Total corner events analyzed: {len(unified_df)}")
                
            else:
                print("No corner analysis data was generated")
        else:
            print("No events processed")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()