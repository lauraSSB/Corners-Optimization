import pandas as pd
import warnings
from typing import List, Dict, Optional, Tuple, Any
from Corner import Corner, corner_zones, _is_in_box
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')

FIELD_WIDTH = 120
FIELD_HEIGHT = 80

def _flip_rect(rect: Tuple[float, float, float, float, str], flip_y: bool = False) -> Tuple[float, float, float, float, str]:
    """Flip rectangle coordinates vertically for Left corners"""
    xmin, xmax, ymin, ymax, name = rect
    if flip_y:
        # Mirror vertically (y-axis flip only)
        ymin, ymax = FIELD_HEIGHT - ymax, FIELD_HEIGHT - ymin
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return (xmin, xmax, ymin, ymax, name)


def corner_zones(corner: str = "Right") -> List[Tuple[float, float, float, float, str]]:
    """Predefined corner zones - only Right and Left corners exist in StatsBomb"""
    base_zones = [
        (111, 120, 0, 18, "A3"),    # 1
        (102, 111, 0, 18, "A1"),    # 2
        (111, 120, 62, 80, "A4"),   # 3
        (102, 111, 62, 80, "A2"),   # 4
        (81, 102, 0, 18, "B1"),     # 5
        (81, 102, 18, 40, "B2"),    # 6
        (81, 102, 40, 62, "B3"),    # 7
        (81, 102, 62, 80, "B4"),    # 8
        (60, 81, 0, 80, "C1"),      # 9,
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
        (114, 120, 18, 30, "Short return – central"),  # 12
        (114, 120, 50, 62, "Far-post low (6yd, mid-height)"),  # 13
    ]

    # Only Left corners need vertical mirroring
    # Right corners: no flipping needed (zones are already correct)
    # Left corners: flip vertically only (y-axis mirror)
    flip_y = corner == "Left"

    return [_flip_rect(r, flip_y=flip_y) for r in base_zones]

class CornerNextBallLocations:
    """Handles five next moments analysis: immediate ball receipts after corners"""

    def __init__(self, downloader: DataDownloader = None):
        self.downloader = downloader or DataDownloader()
        self.next_ball_areas = Dict

    def find_next_ball_receipts(self, processed_events_df: pd.DataFrame, match_events: pd.DataFrame) -> pd.DataFrame:
        """
            Finds the first ball receipt–type events ("Ball Receipt", "Ball Receipt*", "Ball Recovery")
            that follow each corner’s main event (P0) in the same possession.

            Steps:
            1. Get all P0 ids from processed_events_df.
            2. Match them in match_events and infer their side ("Right"/"Left") using Y coordinate.
            3. For each P0, find ball receipt events in the same possession.
            4. Sort by timestamp and keep the first 5.
            5. Send those to get_zones() and combine results.

            Returns
            -------
            pd.DataFrame
                Concatenated results from all P0s (empty if none found).
            """
        try:
            p0_ids = (
                processed_events_df['event_id']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not p0_ids:
                return match_events.iloc[0:0].copy()

            match_events = match_events.copy()
            match_events['id'] = match_events['id'].astype(str)

            p0_rows = (
                match_events.loc[match_events['id'].isin(p0_ids), ['id', 'possession', 'location']]
                .assign(location=lambda df: df['location'].apply(tuple))
                .rename(columns={'id': 'event_id'})
                .drop_duplicates()
            )

            def side_from_location(loc):
                """
                   Determines the attacking side ("Right" or "Left") based on the Y coordinate of a location.
                """
                if isinstance(loc, (list, tuple)) and len(loc) >= 2 and loc[1] is not None:
                    y = loc[1]
                    return "Right" if y < (FIELD_HEIGHT / 2.0) else "Left"
                return "Right"

            p0_rows['side'] = p0_rows['location'].apply(side_from_location)

            if p0_rows.empty:
                return match_events.iloc[0:0].copy()

            ball_receipt_df = []
            ball_receipt_labels = {'Ball Receipt', 'Ball Receipt*', 'Ball Recovery'}

            for _, row in p0_rows.iterrows():
                p0_id = row['event_id']
                poss = row['possession']
                side = row['side']

                receipts = match_events[
                    (match_events['possession'] == poss) &
                    (match_events['type'].isin(ball_receipt_labels))
                    ].copy()

                if receipts.empty:
                    continue

                receipts['side'] = side

                receipts = receipts.sort_values(by=['timestamp'])

                top5 = receipts.head(5).copy()
                if top5.empty:
                    continue

                ball_receipt_df.append(self.get_zones(top5, p0_id))

            final_ball_receipt_df = pd.concat(ball_receipt_df, ignore_index=True) if ball_receipt_df else pd.DataFrame()
            final_ball_receipt_df = final_ball_receipt_df.rename(columns={
                'p0_id': 'event_id'
            })

            return final_ball_receipt_df
        except KeyError as e:
            print(f"KeyError: {e}")
            return match_events.iloc[0:0].copy()


    def get_zones(self, next_events: pd.DataFrame, p0_id: str) -> pd.DataFrame:
        """
            Maps up to five events after a corner (P0) to predefined pitch zones.

            Steps:
            1. Extract (x, y) from 'location'.
            2. Extract side ('Right'/'Left').
            3. Match each event to a zone using corner_zones().
            4. Store zones as M1–M5 linked to the P0 id.

            Returns
            -------
            pd.DataFrame
                One-row DataFrame: ['p0_id', 'M1', 'M2', 'M3', 'M4', 'M5'].
            """
        try:
            events = next_events.copy()

            events[['x', 'y']] = pd.DataFrame(events['location'].tolist(), index=events.index)

            if "side" not in events.columns:
                events["side"] = "Right"

            corner_zones_list = []

            for _, row in events.iterrows():
                side = row["side"]
                x, y = row["x"], row["y"]

                if pd.isna(x) or pd.isna(y):
                    corner_zones_list.append(None)
                    continue

                matched_zone = None
                for xmin, xmax, ymin, ymax, name in corner_zones(side):
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        matched_zone = name
                        break
                corner_zones_list.append(matched_zone)

            events["corner_zone"] = corner_zones_list

            ball_zones: dict[str, str | None] = {"p0_id": str(p0_id)}
            ball_zones.update({f"M{i}": None for i in range(1, 6)})

            if "corner_zone" in events.columns:
                for i, (_, row) in enumerate(events.head(5).iterrows(), start=1):
                    z = row.get("corner_zone")
                    if pd.isna(z):
                        z = None
                    else:
                        z = str(z)
                    ball_zones[f"M{i}"] = z

            cols = ["p0_id", "M1", "M2", "M3", "M4", "M5"]
            return pd.DataFrame([{c: ball_zones.get(c) for c in cols}])

        except KeyError as e:
            print(f"KeyError: {e}")
            return