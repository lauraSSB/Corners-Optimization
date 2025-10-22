import pandas as pd
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from CornerP0 import corner_zones, _is_in_box, FIELD_WIDTH, FIELD_HEIGHT
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')

@dataclass
class CornerP1:
    """
    Handles P1 analysis with robust teammate identification
    Uses the same zone logic as CornerP0 but with perspective correction
    """
    
    # Core identifiers
    p0_event_id: str
    p1_event_id: str
    p0_team: str  # Corner-taking team (always the attacker perspective)
    p1_team: str  # Team that performed the P1 action
    
    # Event details
    match_id: int
    p1_type: str
    p1_timestamp: str
    p1_index: int
    
    # Freeze frame data (already adjusted to P0 perspective)
    freeze_frame: List[Dict[str, Any]]
    
    # Analysis results
    defenders_in_18yd_box: int = field(init=False, default=0)
    attackers_in_6yd_box: int = field(init=False, default=0)
    attackers_out_6yd_box: int = field(init=False, default=0)
    zone_counts: Dict[int, Dict[str, int]] = field(init=False, default_factory=dict)
    corner_side: str = field(init=False, default="Right")
    gk_coordinates: Tuple[Optional[float], Optional[float]] = field(init=False, default=(None, None))

    def __post_init__(self) -> None:
        """Initialize analysis based on adjusted freeze frame"""
        # Determine corner side from P0 perspective (always Right in adjusted coords)
        self.corner_side = "Right"
        
        # Calculate zone counts
        self.zone_counts = self.count_players_in_zones()
        
        # Calculate box counts from zone data
        self.defenders_in_18yd_box = self._count_defenders_in_18yd_box()
        self.attackers_in_6yd_box = self._count_attackers_in_6yd_box()
        self.attackers_out_6yd_box = self._count_attackers_out_6yd_box()
        
        # Get GK coordinates
        self.gk_coordinates = self._get_goalkeeper_coordinates()

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

    def _iter_freeze_players(self):
        """Yield player dicts from freeze_frame"""
        if not self.freeze_frame:
            return
            
        for p in self.freeze_frame:
            if isinstance(p, dict):
                yield p

    def _count_defenders_in_18yd_box(self) -> int:
        """Count defenders in 18-yard box (zones 1-13)"""
        if not self.zone_counts:
            return 0
        
        total_defenders = 0
        for zone_id in range(1, 14):
            if zone_id in self.zone_counts:
                total_defenders += self.zone_counts[zone_id]['defenders']
        
        return total_defenders

    def _count_attackers_in_6yd_box(self) -> int:
        """Count attackers in 6-yard box (zone 2)"""
        if not self.zone_counts or 2 not in self.zone_counts:
            return 0
        return self.zone_counts[2]['attackers']

    def _count_attackers_out_6yd_box(self) -> int:
        """Count attackers outside 6-yard box but in 18-yard box (zones 1, 3-13)"""
        if not self.zone_counts:
            return 0
        
        total_attackers = 0
        for zone_id in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            if zone_id in self.zone_counts:
                total_attackers += self.zone_counts[zone_id]['attackers']
        
        return total_attackers

    def count_players_in_zones(self, zones_definition: Optional[List[Tuple]] = None) -> Dict[int, Dict[str, int]]:
        """
        Count players in predefined zones using the same logic as CornerP0
        """
        if not self.freeze_frame:
            return {}

        # Use zones for Right corners (since we've normalized coordinates)
        zones = zones_definition or corner_zones("Right")

        zones_data = {
            idx: {'attackers': 0, 'defenders': 0, 'total': 0, 'name': name}
            for idx, (xmin, xmax, ymin, ymax, name) in enumerate(zones, start=1)
        }

        for player_data in self._iter_freeze_players():
            loc = player_data.get('location')
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            x, y = loc[0], loc[1]
            
            # Teammate flag should already be correct from perspective adjustment
            is_teammate = self._coerce_bool(player_data.get('teammate', False))
            is_keeper = self._coerce_bool(player_data.get('keeper', False))

            # Check each zone for player presence
            for zone_id, zone_info in zones_data.items():
                xmin, xmax, ymin, ymax, _ = zones[zone_id - 1]
                if _is_in_box(x, y, xmin, xmax, ymin, ymax):
                    zones_data[zone_id]['total'] += 1
                    if is_teammate:
                        zones_data[zone_id]['attackers'] += 1
                    elif not is_keeper:  # Exclude GK from defender counts
                        zones_data[zone_id]['defenders'] += 1
                    break

        return zones_data

    def _get_goalkeeper_coordinates(self) -> Tuple[Optional[float], Optional[float]]:
        """Get goalkeeper coordinates from freeze frame"""
        if not self.freeze_frame:
            return None, None

        for player_data in self.freeze_frame:
            if self._coerce_bool(player_data.get("keeper", False)):
                loc = player_data.get("location")
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    return loc[0], loc[1]
        return None, None

    def zones_to_dict(self) -> Dict[str, Any]:
        """Convert zone counts to flat dictionary for DataFrame integration"""
        result = {}
        for zone_id, data in self.zone_counts.items():
            if zone_id == 2:  # Skip redundant zone
                continue
            result.update({
                f'P1_n_att_zone_{zone_id}': data['attackers'],
                f'P1_n_def_zone_{zone_id}': data['defenders'],
                f'P1_total_n_zone_{zone_id}': data['total']
            })
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert the P1 analysis to a flat dictionary"""
        gk_x, gk_y = self.gk_coordinates

        base_dict = {
            'P0_event_id': self.p0_event_id,
            'P1_event_id': self.p1_event_id,
            'P1_type': self.p1_type,
            'P1_timestamp': self.p1_timestamp,
            'P1_index': self.p1_index,
            'P1_team': self.p1_team,
            'P1_n_defenders_in_18yd_box': self.defenders_in_18yd_box,
            'P1_n_attackers_in_6yd_box': self.attackers_in_6yd_box,
            'P1_n_attackers_out_6yd_box': self.attackers_out_6yd_box,
            'P1_GK_x': gk_x,
            'P1_GK_y': gk_y,
            'P1_coordinates_normalized': self.p0_team != self.p1_team,  # True if we flipped coords
        }

        # Add zone data
        base_dict.update(self.zones_to_dict())

        return base_dict