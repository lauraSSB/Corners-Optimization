import pandas as pd
import warnings
from typing import List, Dict, Optional, Tuple, Any
from Corner import Corner, corner_zones, _is_in_box
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')

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
    
    def _analyze_p1_event(self, p1_event: pd.Series, p0_corner: Corner) -> Dict[str, Any]:
        """
        Analyze a single P1 event - count players in zones and get GK position
        Uses the same logic as Corner class but for P1 timeframe
        """
        match_id = p1_event.get('match_id')
        event_id = p1_event.get('id', '')
        
        try:
            # Get freeze frame data for P1 event (we already verified it exists)
            frames_dict = self.downloader.get_match_frames(match_id)
            freeze_frame = frames_dict.get(event_id, [])
            
            if not freeze_frame:
                print(f"Warning: Freeze frame not found for P1 event {event_id} (should not happen)")
                return None
            
            # Use the same corner side as P0
            corner_side = p0_corner.corner_side
            zones = corner_zones(corner_side)
            
            # Initialize zone counts (same structure as Corner class)
            zones_data = {
                idx: {'attackers': 0, 'defenders': 0, 'total': 0, 'name': name}
                for idx, (xmin, xmax, ymin, ymax, name) in enumerate(zones, start=1)
            }
            
            # Count players in zones for P1
            gk_x, gk_y = None, None
            
            # Flatten freeze frame if needed (same logic as Corner._iter_freeze_players)
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
                
                # Store GK coordinates
                if is_keeper and gk_x is None:
                    gk_x, gk_y = x, y
                
                # Check each zone for player presence (same logic as Corner.count_players_in_zones)
                for zone_id, zone_info in zones_data.items():
                    xmin, xmax, ymin, ymax, _ = zones[zone_id - 1]
                    if _is_in_box(x, y, xmin, xmax, ymin, ymax):
                        zones_data[zone_id]['total'] += 1
                        if is_teammate:
                            zones_data[zone_id]['attackers'] += 1
                        elif not is_keeper:  # Exclude GK from defender counts
                            zones_data[zone_id]['defenders'] += 1
                        break
            
            # Convert zone counts to flat dictionary (P1 version)
            zone_dict = {}
            for zone_id, data in zones_data.items():
                if zone_id == 2:  # Skip redundant zone (same as Corner.zones_to_dict)
                    continue
                zone_dict.update({
                    f'P1_n_att_zone_{zone_id}': data['attackers'],
                    f'P1_n_def_zone_{zone_id}': data['defenders'],
                    f'P1_total_n_zone_{zone_id}': data['total']
                })
            
            # Count players in boxes for P1 (same logic as Corner methods)
            defenders_in_18yd = self._count_in_box_p1(
                freeze_frame=freeze_frame,
                p0_corner=p0_corner,
                teammate_filter=False,
                x_min=102, x_max=120, y_min=18, y_max=62,
                exclude_goalkeeper=True
            )
            
            attackers_in_6yd = self._count_in_box_p1(
                freeze_frame=freeze_frame,
                p0_corner=p0_corner,
                teammate_filter=True,
                x_min=114, x_max=120, y_min=36, y_max=44
            )
            
            attackers_out_6yd = self._count_out_box_p1(freeze_frame, p0_corner)
            
            return {
                'P1_event_id': event_id,
                'P1_type': p1_event.get('type', ''),
                'P1_timestamp': p1_event.get('timestamp', ''),
                'P1_index': p1_event.get('index', ''),
                'P1_n_defenders_in_18yd_box': defenders_in_18yd,
                'P1_n_attackers_in_6yd_box': attackers_in_6yd,
                'P1_n_attackers_out_6yd_box': attackers_out_6yd,
                'P1_GK_x': gk_x,
                'P1_GK_y': gk_y,
                **zone_dict
            }
            
        except Exception as e:
            print(f"Error analyzing P1 event {event_id}: {str(e)}")
            return None
    
    def _count_in_box_p1(self, freeze_frame: List, p0_corner: Corner,
                        teammate_filter: Optional[bool], 
                        x_min: float, x_max: float, y_min: float, y_max: float,
                        exclude_goalkeeper: bool = True) -> int:
        """
        Count players in box for P1 (same logic as Corner._count_in_box)
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

            if not _is_in_box(x, y, x_min, x_max, y_min, y_max):
                continue

            # Robust flags (same as Corner._coerce_bool)
            is_keeper = p0_corner._coerce_bool(player_data.get("keeper", False))
            is_teammate = p0_corner._coerce_bool(player_data.get("teammate", False))

            # Always honor explicit GK exclusion first
            if exclude_goalkeeper and is_keeper:
                continue
            
            if teammate_filter is None:
                cnt += 1
            else:
                if is_teammate == teammate_filter:
                    cnt += 1

        return cnt
    
    def _count_out_box_p1(self, freeze_frame: List, p0_corner: Corner) -> int:
        """
        Count attackers outside 6-yard but inside 18-yard box for P1
        (same logic as Corner._count_out_box)
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
            is_teammate = p0_corner._coerce_bool(player_data.get("teammate", False))

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


