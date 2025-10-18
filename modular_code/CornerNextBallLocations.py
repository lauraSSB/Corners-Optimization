import pandas as pd
import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from Corner import Corner, corner_zones, _is_in_box
from DataDownloader import DataDownloader

warnings.filterwarnings('ignore')

FIELD_WIDTH = 120
FIELD_HEIGHT = 80

def _flip_rect(rect: Tuple[float, float, float, float, str],
               flip_y: bool = False) -> Tuple[float, float, float, float, str]:
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
        (60, 81, 0, 80, "C1"),      # 9
    ]

    # Only Left corners need vertical mirroring
    # Right corners: no flipping needed (zones are already correct)
    # Left corners: flip vertically only (y-axis mirror)
    flip_y = corner == "Left"

    return [_flip_rect(r, flip_y=flip_y) for r in base_zones]

class CornerNextBallLocations:
    """Handles four next moments analysis: immediate ball receipts after corners"""

    def __init__(self, downloader: DataDownloader = None):
        self.downloader = downloader or DataDownloader()
        self.next_ball_areas = Dict

    def find_next_ball_receipts(self, processed_events_df: pd.DataFrame, match_events: pd.DataFrame) -> Optional[
        List]:
        try:
            #Get P0 ids from all corners in a match
            p0_ids = (
                processed_events_df['event_id']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if not p0_ids:
                return match_events.iloc[0:0].copy()

            #Get all match events with P0 ids
            match_events = match_events.copy()
            match_events['id'] = match_events['id'].astype(str)

            p0_rows = (
                match_events.loc[match_events['id'].isin(p0_ids), ['id', 'possession']]
                .rename(columns={'id': 'event_id'})
                .dropna()
                .drop_duplicates()
            )

            if p0_rows.empty:
                return match_events.iloc[0:0].copy()

            #Get all the events related with the number of possesion of the P0 events. Filtered by Ball Receipt
            results = []

            ball_receipt_labels = {'Ball Receipt', 'Ball Receipt*'}
            for _, row in p0_rows.iterrows():
                p0_id = row['event_id']
                poss = row['possession']

                print(p0_id, poss)

                receipts = match_events[
                    (match_events['possession'] == poss) &
                    (match_events['type'].isin(ball_receipt_labels))
                    ].copy()

                if receipts.empty:
                    continue


                receipts = receipts.sort_values(by=['timestamp'])

                top4 = receipts.head(4).copy()
                if top4.empty:
                    continue

                print("TOP 4: ",top4)

                self.get_zones(top4)


                if not receipts.empty:
                    receipts['source_event_id'] = p0_id
                    results.append(receipts)

            if not results:
                return match_events.iloc[0:0].copy()

            result = pd.concat(results, ignore_index=True)

            # Orden opcional
            if 'index' in result.columns:
                result = result.sort_values('index').reset_index(drop=True)
            elif {'minute', 'second'}.issubset(result.columns):
                result = result.sort_values(['minute', 'second']).reset_index(drop=True)

            return result
        except KeyError as e:
            print(f"KeyError: {e}")
            return match_events.iloc[0:0].copy()


    def get_zones(self, next_events: pd.DataFrame) -> Dict:
        try:
            events = next_events.copy()

            events[['x', 'y']] = pd.DataFrame(events['location'].tolist(), index=events.index)

            print("CORNER1 ", events)

            ball_zones = {f"M{i}": None for i in range(1, 5)}

            zones = corner_zones("Right")
            masks = []
            labels = []

            for xmin, xmax, ymin, ymax, name in zones:
                masks.append(
                    (events['x'] >= xmin) & (events['x'] <= xmax) &
                    (events['y'] >= ymin) & (events['y'] <= ymax)
                )
                labels.append(name)

            events['corner_zone'] = np.select(masks, labels, default=None)
            print("CORNER2 ",events)

            # AHORA: cada fila ya "pertenece" a la zona que quedó en events['corner_zone'].
            # Si quieres el diccionario M1..M4 para las primeras 4 filas:
            ball_zones = {f"M{i}": None for i in range(1, 5)}
            for i, (_, row) in enumerate(events.head(4).iterrows(), start=1):
                ball_zones[f"M{i}"] = row['corner_zone']
        except KeyError as e:
            print(f"KeyError: {e}")
