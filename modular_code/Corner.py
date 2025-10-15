import pandas as pd
import warnings
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache

warnings.filterwarnings('ignore')
load_dotenv()

# --- StatsBomb pitch geometry (x: 0→120, y: 0→80) ---
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
EIGHTEEN_YRDS_BOX_X_MIN = 102
EIGHTEEN_YRDS_BOX_X_MAX = 120
EIGHTEEN_YRDS_BOX_Y_MIN = 18
EIGHTEEN_YRDS_BOX_Y_MAX = 62

# 6-yard box coordinates
SIX_YARD_X_MIN = 114
SIX_YARD_X_MAX = 120
SIX_YARD_Y_MIN = 36
SIX_YARD_Y_MAX = 44

def _flip_rect(rect: Tuple[float, float, float, float, str],
               flip_y: bool = False) -> Tuple[float, float, float, float, str]:
    """Flip rectangle coordinates vertically for BR corners"""
    xmin, xmax, ymin, ymax, name = rect
    if flip_y:
        # Mirror vertically (y-axis flip only)
        ymin, ymax = FIELD_HEIGHT - ymax, FIELD_HEIGHT - ymin
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return (xmin, xmax, ymin, ymax, name)

@lru_cache(maxsize=2)  # Only need TR and BR zones
def corner_zones(corner: str = "TR") -> List[Tuple[float, float, float, float, str]]:
    """Predefined corner zones - only TR and BR corners exist in StatsBomb"""
    base_zones = [
        (114, 120, 30, 36, "Near-post low (6yd)"),    # 1
        (114, 120, 36, 44, "Central (6yd/GK)"),    # 2
        (114, 120, 44, 50, "Far-post high (6yd)"),    # 3
        (108, 114, 18, 30, "Near-post channel (outside 6yd)"),  # 4
        (108, 114, 30, 36, "Central corridor (outside 6-yard)"), # 5
        (108, 114, 36, 44, "Far-post corridor (outside 6yd)"),   # 6
        (108, 114, 44, 50, "Deep central channel (penalty spot depth)"), # 7
        (108, 114, 50, 62, "Far-post high channel (deep back-post)"),    # 8
        (102, 108, 30, 50, "Penalty-spot corridor"),    # 9
        (102, 108, 18, 30, "Short return – near"),    # 10
        (102, 108, 50, 62, "Short return – far"),    # 11
        (114, 120, 18, 30, "Short return – central"),   # 12
        (114, 120, 50, 62, "Far-post low (6yd, mid-height)"),    # 13
        (102, 120, 0, 18, "Advanced wide return (touchline channel)"),   # 14
    ]

    # Only BR corners need vertical mirroring
    # TR corners: no flipping needed (zones are already correct)
    # BR corners: flip vertically only (y-axis mirror)
    flip_y = corner == "BR"

    return [_flip_rect(r, flip_y=flip_y) for r in base_zones]

def _is_in_box(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
    """Check if coordinates are within a rectangular box"""
    return x_min<= x <= x_max and y_min <= y <= y_max

@dataclass
class Corner:
    '''Optimized Corner class with proper zone mirroring for TR/BR corners only'''

    # Raw event fields
    match_id: int
    event_id: int
    team: str
    player: str
    minute: int
    second: int
    period: int
    location: Optional[Tuple[float, float]]
    end_location: Optional[Tuple[float, float]]
    pass_outcome: Optional[str]
    pass_height: Optional[int]
    pass_length: Optional[int]
    pass_technique: Optional[str]
    pass_type: Optional[str]
    body_part: Optional[str]
    play_pattern: Optional[str]
    recipient: Optional[str]
    related_events: Optional[List[Any]]
    freeze_frame: Optional[List[Dict[str, Any]]]

    # Zone counting fields (auto-filled in __post_init__)
    defenders_in_18yd_box: int = field(init=False, default=0)
    attackers_in_6yd_box: int = field(init=False, default=0)
    attackers_out_6yd_box: int = field(init=False, default=0)

    zone_counts: Dict[int, Dict[str, int]] = field(init=False, default_factory=dict)
    corner_side: str = field(init=False, default="TR")

    def __post_init__(self) -> None:
        """Initialize derived attributes"""
        # Determine actual corner side (only TR or BR exist)
        self.corner_side = self._determine_corner_side()

        self.defenders_in_18yd_box = self._count_in_box(
            teammate_filter=False,
            x_min=EIGHTEEN_YRDS_BOX_X_MIN, x_max=EIGHTEEN_YRDS_BOX_X_MAX,
            y_min=EIGHTEEN_YRDS_BOX_Y_MIN, y_max=EIGHTEEN_YRDS_BOX_Y_MAX,
            exclude_goalkeeper=True
        )

        self.attackers_in_6yd_box = self._count_in_box(
            teammate_filter=True,
            x_min=SIX_YARD_X_MIN, x_max=SIX_YARD_X_MAX,
            y_min=SIX_YARD_Y_MIN, y_max=SIX_YARD_Y_MAX
        )

        self.attackers_out_6yd_box = self._count_out_box()

        # Zone counting - mirror zones for BR corners only
        self.zone_counts = self.count_players_in_zones()

    def _determine_corner_side(self) -> str:
        """
        Determines from which corner the corner is executed
        Only TR or BR corners exist in StatsBomb (attacking left-to-right)
        """
        if not self.location or self.location[0] is None:
            return "TR"

        x, y = self.location

        # StatsBomb always frames with attacking team left-to-right
        # So corners are only from top-right (TR) or bottom-right (BR)
        is_top = y > FIELD_HEIGHT / 2  # Use FIELD_HEIGHT for y-coordinate comparison

        # Since attacking team is always left-to-right, all corners are on the right side
        # We only need to distinguish between top and bottom
        return "Left" if is_top else "Right"

    def _count_in_box(
        self,
        teammate_filter: Optional[bool],
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        exclude_goalkeeper: bool = True
    ) -> int:
        """
        Count players inside [x_min,x_max] × [y_min,y_max].
        - teammate_filter=True  -> attackers only
        - teammate_filter=False -> defenders only
        - teammate_filter=None  -> everyone
        - exclude_goalkeeper    -> exclude any player with keeper=True from ALL counts
        """
        if not self.freeze_frame:
            return 0

        cnt = 0
        for player_data in self.freeze_frame:
            loc = player_data.get("location", None)
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            x, y = float(loc[0]), float(loc[1])
            if not _is_in_box(x, y, x_min, x_max, y_min, y_max):
                continue

            # Exclude goalkeeper from defender counts
            if exclude_goalkeeper and player_data.get("keeper", False):
                continue

            if teammate_filter is None:
                cnt += 1
            else:
                if bool(player_data.get("teammate", False)) == teammate_filter:
                    cnt += 1

        return cnt

    def _count_out_box(self) -> int:
        """
        Count attackers who are outside the 6-yard box but inside the 18-yard box
        """
        if not self.freeze_frame:
            return 0

        cnt = 0
        for player_data in self.freeze_frame:
            loc = player_data.get("location", None)
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            x, y = loc[0], loc[1]
            is_teammate = bool(player_data.get("teammate", False))

            # Count attackers who are in 18-yard box but NOT in 6-yard box
            if (is_teammate and
                _is_in_box(x, y, EIGHTEEN_YRDS_BOX_X_MIN, EIGHTEEN_YRDS_BOX_X_MAX,
                          EIGHTEEN_YRDS_BOX_Y_MIN, EIGHTEEN_YRDS_BOX_Y_MAX) and
                not _is_in_box(x, y, SIX_YARD_X_MIN, SIX_YARD_X_MAX,
                              SIX_YARD_Y_MIN, SIX_YARD_Y_MAX)):
                cnt += 1

        return cnt

    def count_players_in_zones(self, zones_definition: Optional[List[Tuple]] = None) -> Dict[int, Dict[str, int]]:
        """
        Count players in predefined zones
        - TR corners: use standard zones
        - BR corners: mirror zones vertically
        """
        if not self.freeze_frame:
            return {}

        # Use zones with vertical mirroring for BR corners only
        zones = zones_definition or corner_zones(self.corner_side)

        zones_data = {
            idx: {'attackers': 0, 'defenders': 0, 'total': 0, 'name': name}
            for idx, (xmin, xmax, ymin, ymax, name) in enumerate(zones, start=1)
        }

        for player_data in self.freeze_frame:
            loc = player_data.get('location')
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            x, y = loc[0], loc[1]
            is_teammate = player_data.get('teammate', False)
            is_keeper = player_data.get('keeper', False)

            # Check each zone for player presence
            for zone_id, zone_info in zones_data.items():
                zone_coords = zones[zone_id - 1]  # Get (possibly mirrored) coordinates
                xmin, xmax, ymin, ymax, _ = zone_coords

                if _is_in_box(x, y, xmin, xmax, ymin, ymax):
                    zones_data[zone_id]['total'] += 1
                    if is_teammate:
                        zones_data[zone_id]['attackers'] += 1
                    elif not is_keeper:  # Exclude goalkeeper from defender counts
                        zones_data[zone_id]['defenders'] += 1
                    break  # Player can only be in one zone

        return zones_data

    def zones_to_dict(self) -> Dict[str, Any]:
        """Convert zone counts to flat dictionary for DataFrame integration"""
        result = {}
        for zone_id, data in self.zone_counts.items():
            # Skip zone 2 (Central 6yd/GK) as it's redundant with 6-yard box counts
            if zone_id == 2:
                continue
            result.update({
                f'zone_{zone_id}_name': data['name'],
                f'P0_n_att_zone_{zone_id}': data['attackers'],
                f'P0_n_def_zone_{zone_id}': data['defenders'],
                f'P0_total_n_zone_{zone_id}': data['total']
            })
        return result

    def _get_goalkeeper_coordinates(self) -> Tuple[Optional[float], Optional[float]]:
        """Get goalkeeper coordinates from freeze frame"""
        if not self.freeze_frame:
            return None, None

        for player_data in self.freeze_frame:
            if player_data.get("keeper", True):
                loc = player_data.get("location")
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    return loc[0], loc[1]
        return None, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the corner object into a flat dictionary for DataFrames"""
        # Get goalkeeper coordinates
        gk_x, gk_y = self._get_goalkeeper_coordinates()

        base_dict = {
            # Core event data
            "match_id": self.match_id,
            "event_id": self.event_id,
            "team": self.team or '',
            "player": self.player or '',
            "minute": self.minute or 0,
            "second": self.second or 0,
            "period": self.period or 1,
            "pass_outcome": self.pass_outcome or '',
            "pass_height": self.pass_height or '',
            "pass_length": self.pass_length or '',
            "pass_technique": self.pass_technique or '',
            "pass_type": self.pass_type or '',
            "body_part": self.body_part or '',
            "play_pattern": self.play_pattern or '',
            "recipient": self.recipient or '',

            # Location data
            "location_x": self.location[0] if self.location and len(self.location) >= 1 else None,
            "location_y": self.location[1] if self.location and len(self.location) >= 2 else None,
            "end_location_x": self.end_location[0] if self.end_location and len(self.end_location) >= 1 else None,
            "end_location_y": self.end_location[1] if self.end_location and len(self.end_location) >= 2 else None,

            # Box counts
            "P0_n_defenders_in_18yd_box": self.defenders_in_18yd_box,
            "P0_n_attackers_in_6yd_box": self.attackers_in_6yd_box,
            "P0_n_attackers_out_6yd_box": self.attackers_out_6yd_box,

            # Goalkeeper coordinates
            "P0_GK_x": gk_x,
            "P0_GK_y": gk_y,

            # Corner side (only TR or BR)
            "corner_side": self.corner_side,
        }

        # Add zone data
        base_dict.update(self.zones_to_dict())

        return base_dict