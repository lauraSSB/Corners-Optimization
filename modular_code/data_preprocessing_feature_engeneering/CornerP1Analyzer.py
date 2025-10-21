import pandas as pd
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Any
from Corner import Corner, corner_zones, _is_in_box
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')

FIELD_WIDTH = 120
FIELD_HEIGHT = 80

class CornerP1Analyzer:
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
        CORRECTED: Adjust freeze frame to be from P0 perspective by:
        1. Normalizing coordinates to always be from corner-taking team's attacking perspective
        2. Teammate flags should represent: True=attacker, False=defender from P0's perspective
        """
        adjusted_frame = []
    
        for player_data in self._iter_freeze_players(freeze_frame):
            # Create a copy of the player data
            player_copy = player_data.copy()
        
            # STEP 1: Always normalize coordinates to P0's attacking perspective
            loc = player_data.get('location')
            if loc and len(loc) >= 2:
                x, y = loc[0], loc[1]
                x_norm, y_norm = self._normalize_coordinates(x, y, p0_team, p1_team)
                player_copy['location'] = [x_norm, y_norm]
        
            # STEP 2: Determine correct teammate flag for P0 perspective
            is_teammate_p1 = self._coerce_bool(player_data.get('teammate', False))
        
            # The key insight: Teammate flag should always mean "on the corner-taking team"
            # regardless of which team has possession at P1
            if p0_team == p1_team:
                # Same team at P1: P1 teammate flags are already correct
                # Teammate=True means on corner-taking team (attacker)
                # Teammate=False means on defending team
                player_copy['teammate'] = is_teammate_p1
            else:
                # Different team at P1: P1 teammate flags are from P1 team's perspective
                # We need to convert to P0 (corner-taking team) perspective
                # If player is teammate of P1 team, they're defending against P0
                # If player is NOT teammate of P1 team, they're on P0 team (attacking)
                player_copy['teammate'] = not is_teammate_p1
        
            adjusted_frame.append(player_copy)
    
        return adjusted_frame

    def _find_first_p1_with_freeze_frame(self, processed_events_df: pd.DataFrame, p0_event: pd.Series) -> Optional[pd.Series]:
        """
        Find the FIRST P1 event with freeze frame for a given P0 corner event
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
    
    def _analyze_p1_event(self, p1_event: pd.Series, p0_corner: Corner) -> Dict[str, Any]:
        """
        Analyze a single P1 event using the exact same Corner class logic as P0
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

            # Create a temporary Corner object with ADJUSTED freeze frame
            # This uses the exact same logic as CornerAnalyzer for P0
            temp_corner = Corner(
                match_id=p0_corner.match_id,
                event_id=p0_corner.event_id + "_P1",
                team=p0_corner.team,  # Keep P0 team
                player=p0_corner.player,
                minute=p0_corner.minute,
                second=p0_corner.second,
                period=p0_corner.period,
                location=p0_corner.location,
                end_location=p0_corner.end_location,
                pass_outcome=p0_corner.pass_outcome,
                pass_height=p0_corner.pass_height,
                pass_length=p0_corner.pass_length,
                pass_technique=p0_corner.pass_technique,
                pass_type=p0_corner.pass_type,
                body_part=p0_corner.body_part,
                play_pattern=p0_corner.play_pattern,
                recipient=p0_corner.recipient,
                related_events=p0_corner.related_events,
                freeze_frame=adjusted_freeze_frame  # Use ADJUSTED freeze frame
            )

            # Now we have zone counts using the EXACT SAME LOGIC as P0
            zones_data = temp_corner.zone_counts
            
            print(f"\nP1 ZONE COUNTS (using P0 Corner class logic):")
            for zone_id in range(1, 15):
                if zone_id in zones_data:
                    data = zones_data[zone_id]
                    print(f"  Zone {zone_id}: {data['attackers']} attackers, {data['defenders']} defenders")

            # Get GK coordinates from the temporary corner
            gk_x, gk_y = temp_corner._get_goalkeeper_coordinates()

            # Convert zone counts to flat dictionary (P1 version)
            zone_dict = {}
            for zone_id, data in zones_data.items():
                if zone_id == 2:  # Skip redundant zone
                    continue
                zone_dict.update({
                    f'P1_n_att_zone_{zone_id}': data['attackers'],
                    f'P1_n_def_zone_{zone_id}': data['defenders'],
                    f'P1_total_n_zone_{zone_id}': data['total']
                })

            # Use the temporary corner's calculated box counts
            defenders_18yd = temp_corner.defenders_in_18yd_box
            attackers_6yd = temp_corner.attackers_in_6yd_box
            attackers_out_6yd = temp_corner.attackers_out_6yd_box

            print(f"P1 FINAL COUNTS: {defenders_18yd} defenders in 18yd, {attackers_6yd} attackers in 6yd, {attackers_out_6yd} attackers out 6yd")

            return {
                'P1_event_id': event_id,
                'P1_type': p1_event.get('type', ''),
                'P1_timestamp': p1_event.get('timestamp', ''),
                'P1_index': p1_event.get('index', ''),
                'P1_team': p1_team,
                'P1_n_defenders_in_18yd_box': defenders_18yd,
                'P1_n_attackers_in_6yd_box': attackers_6yd,
                'P1_n_attackers_out_6yd_box': attackers_out_6yd,
                'P1_GK_x': gk_x,
                'P1_GK_y': gk_y,
                'P1_coordinates_normalized': p0_team != p1_team,
                **zone_dict
            }
        
        except Exception as e:
            print(f"Error analyzing P1 event {event_id}: {str(e)}")
            traceback.print_exc()
            return None

    def _count_defenders_in_18yd_from_zones(self, zones_data: Dict) -> int:
        """Count defenders in 18-yard box from zone data (zones 1-13) - CONSISTENT with P0"""
        total_defenders = 0
        # Zones 1-13 are within the 18-yard box (zone 14 is outside)
        for zone_id in range(1, 14):
            if zone_id in zones_data:
                total_defenders += zones_data[zone_id]['defenders']
        return total_defenders

    def _count_attackers_in_6yd_from_zones(self, zones_data: Dict) -> int:
        """Count attackers in 6-yard box from zone data (zone 2) - CONSISTENT with P0"""
        if 2 not in zones_data:
            return 0
        return zones_data[2]['attackers']

    def _count_attackers_out_6yd_from_zones(self, zones_data: Dict) -> int:
        """Count attackers outside 6-yard box but in 18-yard box from zone data (zones 1, 3-13) - CONSISTENT with P0"""
        total_attackers = 0
        # All zones except zone 2 (6-yard box) and zone 14 (outside 18-yard box)
        for zone_id in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            if zone_id in zones_data:
                total_attackers += zones_data[zone_id]['attackers']
        return total_attackers

    def validate_p1_vs_p0(self, p0_corners: List[Corner], p1_df: pd.DataFrame):
        """Validate P1 analysis against P0 baseline"""
        print("\n" + "="*50)
        print("P1 vs P0 VALIDATION")
        print("="*50)
        
        validation_results = []
        
        for corner in p0_corners:
            p1_data = p1_df[p1_df['P0_event_id'] == corner.event_id]
            
            if p1_data.empty:
                print(f"Corner {corner.event_id}: No P1 data found")
                validation_results.append({
                    'event_id': corner.event_id,
                    'status': 'NO_P1_DATA',
                    'p0_defenders': corner.defenders_in_18yd_box,
                    'p1_defenders': None,
                    'p0_attackers_6yd': corner.attackers_in_6yd_box,
                    'p1_attackers_6yd': None,
                    'difference_defenders': None,
                    'difference_attackers_6yd': None
                })
                continue
                
            p1_row = p1_data.iloc[0]
            
            p1_defenders = p1_row.get('P1_n_defenders_in_18yd_box', 0)
            p1_attackers_6yd = p1_row.get('P1_n_attackers_in_6yd_box', 0)
            
            diff_defenders = p1_defenders - corner.defenders_in_18yd_box
            diff_attackers_6yd = p1_attackers_6yd - corner.attackers_in_6yd_box
            
            status = "MATCH" if abs(diff_defenders) <= 2 and abs(diff_attackers_6yd) <= 1 else "MISMATCH"
            
            print(f"\nCorner {corner.event_id}: {status}")
            print(f"  Defenders in 18yd: P0={corner.defenders_in_18yd_box}, P1={p1_defenders}, Diff={diff_defenders}")
            print(f"  Attackers in 6yd:  P0={corner.attackers_in_6yd_box}, P1={p1_attackers_6yd}, Diff={diff_attackers_6yd}")
            
            validation_results.append({
                'event_id': corner.event_id,
                'status': status,
                'p0_defenders': corner.defenders_in_18yd_box,
                'p1_defenders': p1_defenders,
                'p0_attackers_6yd': corner.attackers_in_6yd_box,
                'p1_attackers_6yd': p1_attackers_6yd,
                'difference_defenders': diff_defenders,
                'difference_attackers_6yd': diff_attackers_6yd
            })
        
        # Summary statistics
        if validation_results:
            total_corners = len(validation_results)
            matches = sum(1 for r in validation_results if r['status'] == 'MATCH')
            mismatches = sum(1 for r in validation_results if r['status'] == 'MISMATCH')
            no_data = sum(1 for r in validation_results if r['status'] == 'NO_P1_DATA')
            
            print(f"\nSUMMARY:")
            print(f"Total corners: {total_corners}")
            print(f"Matches: {matches} ({matches/total_corners*100:.1f}%)")
            print(f"Mismatches: {mismatches} ({mismatches/total_corners*100:.1f}%)")
            print(f"No P1 data: {no_data} ({no_data/total_corners*100:.1f}%)")
        
        return pd.DataFrame(validation_results)
    
    def analyze_p1_events(self, processed_events_df: pd.DataFrame, p0_corners: List[Corner]) -> pd.DataFrame:
        """
        Analyze P1 events for all given P0 corners
        
        Args:
            processed_events_df: Full processed events DataFrame (ALL events, not just corners)
            p0_corners: List of Corner objects from original analysis
            
        Returns:
            DataFrame with P1 analysis linked to P0 events
        """
        if processed_events_df.empty or not p0_corners:
            print("No processed events or P0 corners to analyze")
            return pd.DataFrame()
        
        print(f"Analyzing P1 events for {len(p0_corners)} P0 corners...")
        
        self.p1_analysis_data = []
        
        # Create a mapping from event_id to Corner object for quick lookup
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
                    
                    # Analyze the P1 event using the simplified approach
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
            
            # Run validation
            self.validate_p1_vs_p0(p0_corners, p1_df)
            
            return p1_df
        else:
            print("No P1 data was successfully processed")
            return pd.DataFrame()

    def merge_p0_p1_data(self, p0_df: pd.DataFrame, p1_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge P0 and P1 DataFrames
        
        Args:
            p0_df: DataFrame from CornerAnalyzer (P0 data)
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