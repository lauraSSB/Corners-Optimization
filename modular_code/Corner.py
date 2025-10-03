import pandas as pd
import warnings
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import math

warnings.filterwarnings('ignore')
load_dotenv()

# --- StatsBomb pitch & box geometry (x: 0→120, y: 0→80) ---
# 18-yard box at attacking end:
PENALTY_X_MIN = 102
PENALTY_X_MAX = 120
PENALTY_Y_MIN = 18
PENALTY_Y_MAX = 62

# 6-yard box at attacking end (updated coordinates):
SIX_YARD_X_MIN = 114
SIX_YARD_X_MAX = 120
SIX_YARD_Y_MIN = 36
SIX_YARD_Y_MAX = 44

@dataclass
class Corner:
    '''Class representing a corner kick with all its attributes'''

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
    pass_height: Optional[str]
    pass_type: Optional[str]
    body_part: Optional[str]
    play_pattern: Optional[str]
    under_pressure: Optional[bool]
    recipient: Optional[str]
    related_events: Optional[List[Any]]
    freeze_frame: Optional[List[Dict[str, Any]]]

    # Derived fields (auto-filled in __post_init__)
    distance: Optional[float] = field(init=False, default=None)

    # 18-yard box counts
    players_in_18yd_box: int = field(init=False, default=0)
    attackers_in_18yd_box: int = field(init=False, default=0)
    defenders_in_18yd_box: int = field(init=False, default=0)

    # 6-yard box counts (optional, handy for set-piece analysis)
    players_in_6yd_box: int = field(init=False, default=0)
    attackers_in_6yd_box: int = field(init=False, default=0)
    defenders_in_6yd_box: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Distance using math.hypot
        self.distance = self._calculate_distance()

        # 18-yard box
        self.players_in_18yd_box   = self._count_in_box(teammate_filter=None,
                                                        x_min=PENALTY_X_MIN, x_max=PENALTY_X_MAX,
                                                        y_min=PENALTY_Y_MIN, y_max=PENALTY_Y_MAX)
        self.attackers_in_18yd_box = self._count_in_box(teammate_filter=True,
                                                        x_min=PENALTY_X_MIN, x_max=PENALTY_X_MAX,
                                                        y_min=PENALTY_Y_MIN, y_max=PENALTY_Y_MAX)
        self.defenders_in_18yd_box = self._count_in_box(teammate_filter=False,
                                                        x_min=PENALTY_X_MIN, x_max=PENALTY_X_MAX,
                                                        y_min=PENALTY_Y_MIN, y_max=PENALTY_Y_MAX)

        # 6-yard box (updated with proper coordinates)
        self.players_in_6yd_box    = self._count_in_box(teammate_filter=None,
                                                        x_min=SIX_YARD_X_MIN, x_max=SIX_YARD_X_MAX,
                                                        y_min=SIX_YARD_Y_MIN, y_max=SIX_YARD_Y_MAX)
        self.attackers_in_6yd_box  = self._count_in_box(teammate_filter=True,
                                                        x_min=SIX_YARD_X_MIN, x_max=SIX_YARD_X_MAX,
                                                        y_min=SIX_YARD_Y_MIN, y_max=SIX_YARD_Y_MAX)
        self.defenders_in_6yd_box  = self._count_in_box(teammate_filter=False,
                                                        x_min=SIX_YARD_X_MIN, x_max=SIX_YARD_X_MAX,
                                                        y_min=SIX_YARD_Y_MIN, y_max=SIX_YARD_Y_MAX)

    # ---------- helpers ----------
    def _calculate_distance(self) -> Optional[float]:
        """Calculate Euclidean distance between start and end location."""
        if self.location and self.end_location:
            x1, y1 = self.location
            x2, y2 = self.end_location
            return math.hypot(x2 - x1, y2 - y1)
        return None

    def _count_in_box(
        self,
        teammate_filter: Optional[bool],
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
    ) -> int:
        """
        Generic counter for players inside a rectangular box.
        - teammate_filter = True  → only attackers (teammates)
        - teammate_filter = False → only defenders (opponents)
        - teammate_filter = None  → everyone
        """
        if not self.freeze_frame:
            return 0

        cnt = 0
        for p in self.freeze_frame:
            loc = p.get("location", None)
            if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
                continue

            x, y = loc[0], loc[1]
            # Check if player is within the rectangular box boundaries
            if x_min <= x <= x_max and y_min <= y <= y_max:
                if teammate_filter is None:
                    cnt += 1
                else:
                    is_tm = bool(p.get("teammate", False))
                    if is_tm == teammate_filter:
                        cnt += 1
        return cnt

    def flip_rect(self, rect, flip_x=False, flip_y=False) -> tuple[int | Any, int | Any, int | Any, int | Any, Any]:
        """Flip a rectangle according to the side of the corner"""
        xmin, xmax, ymin, ymax, name = rect
        if flip_x:
            xmin, xmax = 120 - xmax, 120 - xmin
        if flip_y:
            ymin, ymax = 80 - ymax, 80 - ymin
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        return (xmin, xmax, ymin, ymax, name)

    def corner_zones(self, corner="TR") -> list:
        """Definition of all interest areas"""
        base = [
            (PENALTY_X_MIN, PENALTY_X_MAX,PENALTY_Y_MIN, PENALTY_Y_MAX, "18yd-box"),
            (SIX_YARD_X_MIN, SIX_YARD_X_MAX, SIX_YARD_Y_MIN, SIX_YARD_Y_MAX, "6yd-box"),
            (114, 120, 30, 36, "Near-post low (6yd)"),  # 1
            (114, 120, 36, 44, "Central (6yd/GK)"),  # 2
            (114, 120, 44, 50, "Far-post high (6yd)"),  # 3
            (108, 114, 18, 30, "Near-post channel (outside 6yd)"),  # 4
            (108, 114, 30, 36, "Central corridor (outside 6-yard)"),  # 5
            (108, 114, 36, 44, "Far-post corridor (outside 6yd)"),  # 6
            (108, 114, 44, 50, "Deep central channel (penalty spot depth)"),  # 7
            (108, 114, 50, 62, "Far-post high channel (deep back-post)"),  # 8
            (102, 108, 30, 50, "Penalty-spot corridor"),  # 9
            (102, 108, 18, 30, "Short return – near"),  # 10
            (102, 108, 50, 62, "Short return – far"),  # 11
            (114, 120, 30, 18, "Short return – central"),  # 12
            (114, 120, 50, 62, "Far-post low (6yd, mid-height)"),  # 13
            (102, 120, 0, 18, "Advanced wide return (touchline channel)"),  # 14
        ]
        flip_x = corner in {"TL", "BL"}
        flip_y = corner in {"BR", "BL"}
        return [self.flip_rect(r, flip_x=flip_x, flip_y=flip_y) for r in base]

    def count_players_in_zones(self, zones_definition) -> dict:
        if not self.freeze_frame:
            return {}

        zones_data = {}
        for idx, (xmin, xmax, ymin, ymax, name) in enumerate(zones_definition, start=1):
            zones_data[idx] = {
                'attackers': self._count_in_box(True, x_min = xmin, x_max = xmax, y_min = ymin, y_max = ymax),
                'defenders': self._count_in_box(False, x_min = xmin, x_max = xmax, y_min = ymin, y_max = ymax),
                'total': self._count_in_box(None, x_min = xmin, x_max = xmax, y_min = ymin, y_max = ymax),
                'name': name,
                'zone_coords': (xmin, xmax, ymin, ymax),
            }
            print(zones_data[idx])

        return zones_data

    def _determine_corner_side(self):
        """
        Determina desde qué esquina se ejecuta el corner basado en location
        Returns: "TR", "TL", "BR", "BL"
        """
        if not self.location or self.location[0] is None:
            return "TR"

        x, y = self.location

        is_top = x > 60  # Mitad superior del campo
        is_right = y > 40  # Mitad derecha del campo

        if is_top and is_right:
            return "TR"
        elif is_top and not is_right:
            return "TL"
        elif not is_top and is_right:
            return "BR"
        else:
            return "BL"

    def _count_players_by_zone(self):
        """
        Cuenta jugadores usando las zonas predefinidas de corner_zones()
        """
        corner_side = self._determine_corner_side()
        zones = self.corner_zones(corner_side)
        return self.count_players_in_zones(zones)

    def zones_to_dict(self, zones_definition=None):
        """
        Convierte los datos de zonas a diccionario plano para to_dict()

        Args:
            zones_definition: Lista de zonas personalizada (opcional)

        Returns:
            dict: Datos aplanados para agregar al to_dict()
        """
        if zones_definition:
            zones_data = self.count_players_in_zones(zones_definition)
        else:
            zones_data = self._count_players_by_zone()

        result = {}
        for zone_num, data in zones_data.items():
            result[f'zone_{zone_num}_attackers'] = data['attackers']
            result[f'zone_{zone_num}_defenders'] = data['defenders']
            result[f'zone_{zone_num}_total'] = data['total']
            result[f'zone_{zone_num}_name'] = data['name']

        return result


    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        """Convert the corner object into a flat dictionary (appropriate for DataFrames)."""
        base_dict =  {
            # raw
            "match_id": self.match_id,
            "event_id": self.event_id,
            "team": self.team,
            "player": self.player,
            "minute": self.minute,
            "second": self.second,
            "period": self.period,
            "pass_outcome": self.pass_outcome,
            "pass_height": self.pass_height,
            "pass_type": self.pass_type,
            "body_part": self.body_part,
            "play_pattern": self.play_pattern,
            "under_pressure": self.under_pressure,
            "recipient": self.recipient,
            # locations
            "location_x": self.location[0] if self.location else None,
            "location_y": self.location[1] if self.location else None,
            "end_location_x": self.end_location[0] if self.end_location else None,
            "end_location_y": self.end_location[1] if self.end_location else None,
            # derived
            "distance": self.distance,
            #"players_in_18yd_box": self.players_in_18yd_box,
            #"attackers_in_18yd_box": self.attackers_in_18yd_box,
            #"defenders_in_18yd_box": self.defenders_in_18yd_box,
            #"players_in_6yd_box": self.players_in_6yd_box,
            #"attackers_in_6yd_box": self.attackers_in_6yd_box,
            #"defenders_in_6yd_box": self.defenders_in_6yd_box,
        }

        base_dict.update(self.zones_to_dict())

        return base_dict
