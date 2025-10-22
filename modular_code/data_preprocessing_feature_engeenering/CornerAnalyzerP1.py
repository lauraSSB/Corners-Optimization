import pandas as pd
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Any
from CornerP0 import CornerP0, _is_in_box, FIELD_WIDTH, FIELD_HEIGHT
from CornerP1 import CornerP1
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')

class CornerAnalyzerP1:
    """Handles P1 analysis: immediate events after corners"""
    
    def __init__(self, downloader: DataDownloader = None):
        self.downloader = downloader or DataDownloader()
        self.p1_analysis_data: List[Dict] = []
    
    def _iter_freeze_players(self, freeze_frame):
        """Yield player dicts from freeze_frame regardless of nested shape [[{...}]] or [{...}]."""
        if not freeze_frame:
            return
        
        # Flatten one level if needed: [[{...}, {...}]] -> [{...}, {...}]
        if isinstance(freeze_frame, list) and len(freeze_frame) > 0 and isinstance(freeze_frame[0], list):
            freeze_frame = freeze_frame[0]

        for p in freeze_frame:
            if isinstance(p, dict):
                yield p

    @staticmethod
    def _coerce_bool(v) -> bool:
        """Robust boolean coercion for fields that may be True/False/1/0/'true'/'false'/etc."""
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y", "t")
        return False

    def _debug_p1_analysis(self, freeze_frame: List, p0_team: str, p1_team: str, adjusted_freeze_frame: List):
        """Enhanced debug method to show coordinate and teammate flag transformations"""
        print(f"\n=== P1 DEBUG ANALYSIS ===")
        print(f"P0 Corner team: {p0_team}")
        print(f"P1 Event team: {p1_team}")
        print(f"Teams same: {p0_team == p1_team}")
    
        # Count original players by team
        orig_teammates = 0
        orig_opponents = 0
        for player_data in self._iter_freeze_players(freeze_frame):
            if self._coerce_bool(player_data.get('teammate', False)):
                orig_teammates += 1
            else:
                orig_opponents += 1
    
        # Count adjusted players by team  
        adj_teammates = 0
        adj_opponents = 0
        for player_data in adjusted_freeze_frame:
            if self._coerce_bool(player_data.get('teammate', False)):
                adj_teammates += 1
            else:
                adj_opponents += 1
    
        print(f"\nTEAM COUNTS:")
        print(f"Original: {orig_teammates} teammates, {orig_opponents} opponents")
        print(f"Adjusted: {adj_teammates} teammates (attackers), {adj_opponents} opponents (defenders)")
    
        print(f"\nOriginal P1 Freeze Frame ({len(list(self._iter_freeze_players(freeze_frame)))} players):")
        for i, player_data in enumerate(self._iter_freeze_players(freeze_frame)):
            loc = player_data.get('location')
            if not loc or len(loc) < 2:
                continue
            
            x, y = loc[0], loc[1]
            is_teammate = self._coerce_bool(player_data.get('teammate', False))
            is_keeper = self._coerce_bool(player_data.get('keeper', False))
        
            print(f"  Player {i}: orig({x:.1f}, {y:.1f}) | "
                f"Teammate: {is_teammate}, Keeper: {is_keeper}")

        print(f"\nAdjusted P1 Freeze Frame ({len(adjusted_freeze_frame)} players):")
        for i, player_data in enumerate(adjusted_freeze_frame):
            loc = player_data.get('location')
            if not loc or len(loc) < 2:
                continue
            
            x, y = loc[0], loc[1]
            is_teammate = self._coerce_bool(player_data.get('teammate', False))
            is_keeper = self._coerce_bool(player_data.get('keeper', False))
            role = "ATTACKER" if is_teammate else "DEFENDER"
        
            print(f"  Player {i}: adj({x:.1f}, {y:.1f}) | "
                f"Role: {role}, Keeper: {is_keeper}")

    def _normalize_coordinates(self, x: float, y: float, p0_team: str, p1_team: str) -> Tuple[float, float]:
        """
        Normalize coordinates to always be from the perspective of the corner-taking team
        """
        if p0_team != p1_team:
            # Different team has possession - flip to maintain attacking perspective
            return FIELD_WIDTH - x, FIELD_HEIGHT - y
        else:
            # Same team perspective - no change needed
            return x, y

    def _adjust_freeze_frame_for_p0_perspective(self, freeze_frame: List, p0_team: str, p1_team: str) -> List[Dict]:
        """
        CORRECTED: Adjust freeze frame to be from P0 perspective using goalkeeper-based logic
        with fallback to original logic if goalkeeper identification fails
        """
        adjusted_frame = []
    
        # Step 1: Find all goalkeepers in the freeze frame
        goalkeepers = []
    
        for player_data in self._iter_freeze_players(freeze_frame):
            if self._coerce_bool(player_data.get('keeper', False)):
                goalkeepers.append(player_data)
    
        # If we have exactly one goalkeeper, use their team as defending team
        defending_team_goalkeeper = None
        if len(goalkeepers) == 1:
            defending_team_goalkeeper = self._coerce_bool(goalkeepers[0].get('teammate', False))
            print(f"Found goalkeeper: teammate={defending_team_goalkeeper}")
        elif len(goalkeepers) > 1:
            print(f"Warning: Multiple goalkeepers ({len(goalkeepers)}) found - using fallback logic")
        else:
            print(f"Warning: No goalkeepers found - using fallback logic")
    
        # Step 2: Process all players and set correct teammate flags
        for player_data in self._iter_freeze_players(freeze_frame):
            player_copy = player_data.copy()
        
            # STEP 1: Normalize coordinates to P0's attacking perspective
            loc = player_data.get('location')
            if loc and len(loc) >= 2:
                x, y = loc[0], loc[1]
                x_norm, y_norm = self._normalize_coordinates(x, y, p0_team, p1_team)
                player_copy['location'] = [x_norm, y_norm]
        
            # STEP 2: Determine correct teammate flag
            if defending_team_goalkeeper is not None:
                # Use goalkeeper-based logic
                is_keeper = self._coerce_bool(player_data.get('keeper', False))
                original_teammate = self._coerce_bool(player_data.get('teammate', False))
            
                if is_keeper:
                    # This is a goalkeeper - they should be on the defending team
                    player_copy['teammate'] = False  # Goalkeeper is defender
                else:
                    # Regular player: if they're on the same team as the goalkeeper, they're defenders
                    if original_teammate == defending_team_goalkeeper:
                        player_copy['teammate'] = False  # Defender
                    else:
                        player_copy['teammate'] = True   # Attacker
            else:
                # Fallback to original logic if goalkeeper identification failed
                is_teammate_p1 = self._coerce_bool(player_data.get('teammate', False))
                
                if p0_team == p1_team:
                    # Same team at P1: P1 teammate flags are already correct
                    player_copy['teammate'] = is_teammate_p1
                else:
                    # Different team at P1: flip teammate flags
                    player_copy['teammate'] = not is_teammate_p1
            
            adjusted_frame.append(player_copy)
    
        # Debug output
        attackers = sum(1 for p in adjusted_frame if p.get('teammate') and not p.get('keeper'))
        defenders = sum(1 for p in adjusted_frame if not p.get('teammate') and not p.get('keeper'))
        goalkeepers = sum(1 for p in adjusted_frame if p.get('keeper'))
        print(f"Final counts - Attackers: {attackers}, Defenders: {defenders}, Goalkeepers: {goalkeepers}")
    
        return adjusted_frame

    def _find_first_p1_with_freeze_frame(self, processed_events_df: pd.DataFrame, p0_event: pd.Series) -> Optional[pd.Series]:
        """
        Find the FIRST P1 event with freeze frame for a given P0 corner event
        Uses the same logic as CornerP1Analyzer
        """
        if 'index' not in processed_events_df.columns:
            print("Required column 'index' not found")
            return None
        
        # Get P0 index
        p0_index = p0_event['index']
        
        print(f"Looking for P1 event after P0: index={p0_index}")
        
        # Find the immediate next event (P0_index + 1)
        next_events = processed_events_df[processed_events_df['index'] > p0_index]
        
        if next_events.empty:
            print(f"No events found after P0 index {p0_index}")
            return None
        
        # Sort by index to get events in order
        next_events = next_events.sort_values('index')
        
        # Get the very next event (P1 candidate)
        first_next_event = next_events.iloc[0]
        p1_candidate_index = first_next_event['index']
        p1_candidate_timestamp = first_next_event.get('timestamp', '')
        
        print(f"Immediate next event: index={p1_candidate_index}, timestamp={p1_candidate_timestamp}")
        
        # Get ALL events with this same P1 timestamp (including the first one)
        events_with_p1_timestamp = next_events[next_events['timestamp'] == p1_candidate_timestamp]
        events_with_p1_timestamp = events_with_p1_timestamp.sort_values('index')
        
        print(f"Found {len(events_with_p1_timestamp)} events with P1 timestamp {p1_candidate_timestamp}")
        
        # Check each event in order for freeze frame
        for _, event in events_with_p1_timestamp.iterrows():
            event_id = event.get('id', '')
            match_id = event.get('match_id')
            
            # Check if this event has a freeze frame
            frames_dict = self.downloader.get_match_frames(match_id)
            freeze_frame = frames_dict.get(event_id, [])
            
            if freeze_frame:
                print(f"✓ Found P1 event with freeze frame: index={event['index']}, type={event.get('type', 'Unknown')}, id={event_id}")
                return event
            else:
                print(f"  Event without freeze frame: index={event['index']}, type={event.get('type', 'Unknown')}, id={event_id}")
        
        print(f"✗ No events with freeze frames found at P1 timestamp {p1_candidate_timestamp}")
        return None
    
    def _analyze_p1_event(self, p1_event: pd.Series, p0_corner: CornerP0) -> Dict[str, Any]:
        """
        Analyze a single P1 event using the CornerP1 class
        """
        match_id = p1_event.get('match_id')
        event_id = p1_event.get('id', '')

        try:
            # Get freeze frame data for P1 event
            frames_dict = self.downloader.get_match_frames(match_id)
            freeze_frame = frames_dict.get(event_id, [])
        
            if not freeze_frame:
                print(f"Warning: Freeze frame not found for P1 event {event_id}")
                return None

            print(f"\n=== ANALYZING P1 EVENT {event_id} ===")
            print(f"P0 Corner: {p0_corner.event_id}, Team: {p0_corner.team}")
            print(f"P1 Event: {event_id}, Team: {p1_event.get('team')}, Type: {p1_event.get('type')}")

            # Get teams for perspective adjustment
            p0_team = p0_corner.team  # Attacking team (corner taker)
            p1_team = p1_event.get('team', '')  # Team that performed P1 action

            # Adjust freeze frame to P0 perspective
            adjusted_freeze_frame = self._adjust_freeze_frame_for_p0_perspective(freeze_frame, p0_team, p1_team)

            # DEBUG: Show coordinate and teammate transformations
            self._debug_p1_analysis(freeze_frame, p0_team, p1_team, adjusted_freeze_frame)

            # Create P1 analysis using CornerP1 class
            p1_analysis = CornerP1(
                p0_event_id=p0_corner.event_id,
                p1_event_id=event_id,
                p0_team=p0_team,
                p1_team=p1_team,
                match_id=match_id,
                p1_type=p1_event.get('type', ''),
                p1_timestamp=f"{p1_event.get('minute', 0)}:{p1_event.get('second', 0):02d}",
                p1_index=p1_event.get('index', 0),
                freeze_frame=adjusted_freeze_frame
            )

            # Get the analysis as dictionary
            result = p1_analysis.to_dict()
            print(f"P1 FINAL COUNTS: {result['P1_n_defenders_in_18yd_box']} defenders in 18yd, {result['P1_n_attackers_in_6yd_box']} attackers in 6yd")

            return result
        
        except Exception as e:
            print(f"Error analyzing P1 event {event_id}: {str(e)}")
            traceback.print_exc()
            return None

    def analyze_p1_events(self, processed_events_df: pd.DataFrame, p0_corners: List[CornerP0]) -> pd.DataFrame:
        """
        Analyze P1 events for all given P0 corners
        
        Args:
            processed_events_df: Full processed events DataFrame (ALL events, not just corners)
            p0_corners: List of CornerP0 objects from original analysis
            
        Returns:
            DataFrame with P1 analysis linked to P0 events
        """
        if processed_events_df.empty or not p0_corners:
            print("No processed events or P0 corners to analyze")
            return pd.DataFrame()
        
        print(f"Analyzing P1 events for {len(p0_corners)} P0 corners...")
        
        self.p1_analysis_data = []
        
        # Create a mapping from event_id to CornerP0 object for quick lookup
        p0_corner_map = {corner.event_id: corner for corner in p0_corners}
        
        # Get all P0 events from processed data for reference
        p0_events = processed_events_df[
            (processed_events_df["play_pattern"] == 'From Corner') & 
            (processed_events_df['type'] == 'Pass') & 
            (processed_events_df['pass_type'] == "Corner")
        ]
        
        p1_events_found = 0
        
        # Process each P0 event to find corresponding P1 events
        for _, p0_event in p0_events.iterrows():
            p0_event_id = p0_event.get('id', '')
            p0_index = p0_event.get('index', '')
            
            if p0_event_id not in p0_corner_map:
                continue
                
            try:
                # Find FIRST P1 event with freeze frame for this P0
                p1_event = self._find_first_p1_with_freeze_frame(processed_events_df, p0_event)
                
                if p1_event is not None:
                    p1_events_found += 1
                    
                    # Analyze the P1 event
                    p1_analysis = self._analyze_p1_event(p1_event, p0_corner_map[p0_event_id])
                    if p1_analysis:
                        p1_analysis['P0_event_id'] = p0_event_id  # Link to P0
                        p1_analysis['P0_index'] = p0_index  # Also store P0 index for reference
                        self.p1_analysis_data.append(p1_analysis)
                else:
                    print(f"No P1 event with freeze frame found for P0 corner {p0_event_id} (index {p0_index})")
                
            except Exception as e:
                print(f"Error processing P1 for P0 event {p0_event_id}: {str(e)}")
                continue
        
        # Create DataFrame from P1 analysis
        if self.p1_analysis_data:
            p1_df = pd.DataFrame(self.p1_analysis_data)
            print(f"Successfully analyzed {len(p1_df)} P1 events with freeze frames")
            
            return p1_df
        else:
            print("No P1 data was successfully processed")
            return pd.DataFrame()

    def merge_p0_p1_data(self, p0_df: pd.DataFrame, p1_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge P0 and P1 DataFrames
        
        Args:
            p0_df: DataFrame from CornerAnalyzerP0 (P0 data)
            p1_df: DataFrame from analyze_p1_events (P1 data)
            
        Returns:
            Combined DataFrame with P0 and P1 data
        """
        if p0_df.empty:
            return p1_df if p1_df is not None else pd.DataFrame()
        
        if p1_df is None or p1_df.empty:
            print("No P1 data to merge, returning P0 data only")
            return p0_df
        
        # Merge P1 data with P0 data (each P0 has at most one P1 now)
        combined_df = p0_df.merge(
            p1_df,
            left_on='event_id',
            right_on='P0_event_id',
            how='left'
        )
        
        # Drop the linking column
        if 'P0_event_id' in combined_df.columns:
            combined_df = combined_df.drop(columns=['P0_event_id'])
        
        print(f"Successfully merged P0 and P1 data: {len(combined_df)} corners")
        print(f"Corners with P1 data: {combined_df['P1_event_id'].notna().sum()}")
        
        return combined_df

    def save_to_csv(self, filename: str = 'p1_analysis.csv') -> Optional[str]:
        """Save P1 analysis to CSV file"""
        if self.p1_analysis_data:
            p1_df = pd.DataFrame(self.p1_analysis_data)
            p1_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"P1 analysis saved to: {filename}")
            return filename
        else:
            print("No P1 analysis data to save")
            return None