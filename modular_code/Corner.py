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
            if not (isinstance(loc, (list, tuple)) and len(loc) >= 2:
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

    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        """Convert the corner object into a flat dictionary (appropriate for DataFrames)."""
        return {
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
            "players_in_18yd_box": self.players_in_18yd_box,
            "attackers_in_18yd_box": self.attackers_in_18yd_box,
            "defenders_in_18yd_box": self.defenders_in_18yd_box,
            "players_in_6yd_box": self.players_in_6yd_box,
            "attackers_in_6yd_box": self.attackers_in_6yd_box,
            "defenders_in_6yd_box": self.defenders_in_6yd_box,
        }
