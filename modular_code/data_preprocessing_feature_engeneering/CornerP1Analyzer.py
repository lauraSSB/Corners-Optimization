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
    
    def _find_first_p1_with_freeze_frame(self, processed_events_df: pd.DataFrame, p0_event: pd.Series) -> Optional[pd.Series]:
        """
        Find the FIRST P1 event with freeze frame for a given P0 corner event
        Corrected strategy:
        1. Get immediate next event (P0_index + 1) - this is P1
        2. Get P1 timestamp
        3. Check ALL events with that same P1 timestamp in order
        4. Return the FIRST one that has a freeze frame
        5. Stop if we move to a different timestamp
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
    
    def _normalize_coordinates(self, x: float, y: float, p0_team: str, p1_team: str) -> Tuple[float, float]:
        """
        Normalize coordinates to always be from the perspective of the corner-taking team
        
        Only flip coordinates if P1 possession team is different from P0 corner team
        """
        needs_normalization = p0_team != p1_team
        
        if not needs_normalization:
            # Same possession team - no coordinate adjustment needed
            return x, y
        else:
            # Different possession team - flip coordinates
            return FIELD_WIDTH - x, FIELD_HEIGHT - y
    
    def _analyze_p1_event(self, p1_event: pd.Series, p0_corner: Corner) -> Dict[str, Any]:
        """
        Analyze a single P1 event - count players in zones and get GK position
        Uses coordinate normalization to ensure consistent perspective
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
        
            # Use the same corner side as P0
            corner_side = p0_corner.corner_side
            zones = corner_zones(corner_side)
        
            # Get teams for coordinate normalization
            p0_team = p0_corner.team  # Attacking team (corner taker)
            p1_team = p1_event.get('team', '')  # Team that performed P1 action
        
            print(f"P0 team: {p0_team}, P1 team: {p1_team}, Normalization needed: {p0_team != p1_team}")
        
            # Initialize zone counts
            zones_data = {
                idx: {'attackers': 0, 'defenders': 0, 'total': 0, 'name': name}
                for idx, (xmin, xmax, ymin, ymax, name) in enumerate(zones, start=1)
            }
        
            # Count players in zones for P1
            gk_x, gk_y = None, None
        
            # Flatten freeze frame if needed
            if isinstance(freeze_frame, list) and len(freeze_frame) > 0 and isinstance(freeze_frame[0], list):
                freeze_frame = freeze_frame[0]
        
            for player_data in freeze_frame:
                if not isinstance(player_data, dict):
                    continue
                
                loc = player_data.get('location')
                if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                    continue
            
                x, y = loc[0], loc[1]
                is_teammate = p0_corner._coerce_bool(player_data.get('teammate', False))
                is_keeper = p0_corner._coerce_bool(player_data.get('keeper', False))
            
                # NORMALIZE COORDINATES to attacking team's perspective
                x, y = self._normalize_coordinates(x, y, p0_team, p1_team)
            
                # CRITICAL: DO NOT flip the teammate flag!
                # The freeze frame teammate flag is from P1 team's perspective
                # After coordinate normalization, we need to determine if player is attacker/defender
                # from P0 team's perspective
            
                # Store GK coordinates (in normalized coordinate system)
                if is_keeper and gk_x is None:
                    gk_x, gk_y = x, y
            
                # Determine if player is attacker or defender FROM P0 PERSPECTIVE
                # If P0 and P1 teams are the same: teammate flag is correct
                # If P0 and P1 teams are different: teammate flag needs interpretation
                if p0_team == p1_team:
                    # Same team perspective: teammate means attacker
                    is_attacker = is_teammate
                else:
                    # Different team perspective: teammate means defender (from P0 view)
                    is_attacker = not is_teammate
            
                # Check each zone for player presence
                for zone_id, zone_info in zones_data.items():
                    xmin, xmax, ymin, ymax, _ = zones[zone_id - 1]
                    if _is_in_box(x, y, xmin, xmax, ymin, ymax):
                        zones_data[zone_id]['total'] += 1
                        if is_attacker and not is_keeper:  # Attackers (excluding GK)
                            zones_data[zone_id]['attackers'] += 1
                        elif not is_keeper:  # Defenders (excluding GK)
                            zones_data[zone_id]['defenders'] += 1
                        break
        
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
        
            # Count players in boxes for P1 using NORMALIZED coordinates
            # Fix the counting methods similarly...
        
            return {
                'P1_event_id': event_id,
                'P1_type': p1_event.get('type', ''),
                'P1_timestamp': p1_event.get('timestamp', ''),
                'P1_index': p1_event.get('index', ''),
                'P1_team': p1_team,
                'P1_n_defenders_in_18yd_box': self._count_defenders_in_18yd_p1(freeze_frame, p0_corner, p0_team, p1_team),
                'P1_n_attackers_in_6yd_box': self._count_attackers_in_6yd_p1(freeze_frame, p0_corner, p0_team, p1_team),
                'P1_n_attackers_out_6yd_box': self._count_out_box_p1(freeze_frame, p0_corner, p0_team, p1_team),
                'P1_GK_x': gk_x,
                'P1_GK_y': gk_y,
                'P1_coordinates_normalized': p0_team != p1_team,
                **zone_dict
            }
        
        except Exception as e:
            print(f"Error analyzing P1 event {event_id}: {str(e)}")
            traceback.print_exc()
            return None

    def _count_defenders_in_18yd_p1(self, freeze_frame: List, p0_corner: Corner, p0_team: str, p1_team: str) -> int:
        """Count defenders in 18yd box from P0 perspective"""
        return self._count_in_box_p1(
            freeze_frame=freeze_frame,
            p0_corner=p0_corner,
            p0_team=p0_team,
            p1_team=p1_team,
            count_attackers=False,  # Count defenders
            x_min=102, x_max=120, y_min=18, y_max=62,
            exclude_goalkeeper=True
        )

    def _count_attackers_in_6yd_p1(self, freeze_frame: List, p0_corner: Corner, p0_team: str, p1_team: str) -> int:
        """Count attackers in 6yd box from P0 perspective"""
        return self._count_in_box_p1(
            freeze_frame=freeze_frame,
            p0_corner=p0_corner,
            p0_team=p0_team,
            p1_team=p1_team,
            count_attackers=True,  # Count attackers
            x_min=114, x_max=120, y_min=36, y_max=44
        )
    
    def _count_in_box_p1(self, freeze_frame: List, p0_corner: Corner,
                    p0_team: str, p1_team: str,
                    count_attackers: bool, 
                    x_min: float, x_max: float, y_min: float, y_max: float,
                    exclude_goalkeeper: bool = True) -> int:
        """
        Count players in box for P1 with proper perspective handling
        """
        cnt = 0
    
        # Flatten freeze frame if needed
        if isinstance(freeze_frame, list) and len(freeze_frame) > 0 and isinstance(freeze_frame[0], list):
            freeze_frame = freeze_frame[0]
    
        for player_data in freeze_frame:
            if not isinstance(player_data, dict):
                continue
            
            loc = player_data.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            try:
                x, y = float(loc[0]), float(loc[1])
            except (TypeError, ValueError):
                continue

            # NORMALIZE COORDINATES
            x, y = self._normalize_coordinates(x, y, p0_team, p1_team)

            if not _is_in_box(x, y, x_min, x_max, y_min, y_max):
                continue

            # Robust flags
            is_keeper = p0_corner._coerce_bool(player_data.get("keeper", False))
            is_teammate = p0_corner._coerce_bool(player_data.get("teammate", False))

            # Determine if player is attacker from P0 perspective
            if p0_team == p1_team:
                is_attacker = is_teammate
            else:
                is_attacker = not is_teammate

            # Always honor explicit GK exclusion first
            if exclude_goalkeeper and is_keeper:
                continue
        
            # Count based on requested type
            if count_attackers and is_attacker:
                cnt += 1
            elif not count_attackers and not is_attacker and not is_keeper:
                cnt += 1

        return cnt
    
    def _count_out_box_p1(self, freeze_frame: List, p0_corner: Corner, 
                         p0_team: str, p1_team: str) -> int:
        """
        Count attackers outside 6-yard but inside 18-yard box for P1 with coordinate normalization
        """
        cnt = 0
        
        # Flatten freeze frame if needed
        if isinstance(freeze_frame, list) and len(freeze_frame) > 0 and isinstance(freeze_frame[0], list):
            freeze_frame = freeze_frame[0]
            
        for player_data in freeze_frame:
            if not isinstance(player_data, dict):
                continue
                
            loc = player_data.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            x, y = loc[0], loc[1]
            
            # NORMALIZE COORDINATES
            x, y = self._normalize_coordinates(x, y, p0_team, p1_team)
            
            is_teammate = p0_corner._coerce_bool(player_data.get("teammate", False))
            
            # Adjust teammate flag if perspectives are different
            if p0_team != p1_team:
                is_teammate = not is_teammate

            # Count attackers who are in 18-yard box but NOT in 6-yard box
            if (is_teammate and
                _is_in_box(x, y, 102, 120, 18, 62) and  # 18-yard box
                not _is_in_box(x, y, 114, 120, 36, 44)):  # 6-yard box
                cnt += 1

        return cnt
    
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
        return combined_df
