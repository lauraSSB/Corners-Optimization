# preprocessing.py
import pandas as pd
from typing import Any, Tuple, Dict

EXCLUDE_COLS = ['goal_20s', 'goal_20s_def', 'xg_20s', 'xg_20s_def']
IDENTIFIER = 'event_id'
DROP_COLS = [
    'match_id','minute','second','period','team','player','pass_type','play_pattern',
    'recipient','P0_index_x','P0_index_y','corner_execution_time_raw','match_date',
    'home_team','away_team','season','P1_event_id','P1_index','P1_timestamp',
    'zone_1_name','P0_total_n_zone_1','zone_3_name','P0_total_n_zone_3','zone_4_name','P0_total_n_zone_4',
    'zone_5_name','P0_total_n_zone_5','zone_6_name','P0_total_n_zone_6','zone_7_name','P0_total_n_zone_7',
    'zone_8_name','P0_total_n_zone_8','zone_9_name','P0_total_n_zone_9','zone_10_name','P0_total_n_zone_10',
    'zone_11_name','P0_total_n_zone_11','zone_12_name','P0_total_n_zone_12','zone_13_name','P0_total_n_zone_13',
    'zone_14_name','P0_total_n_zone_14','pass_outcome','P1_type',
    # OJO: quita 'location_x','location_y' de aquí si el modelo los usa
    # 'location_x','location_y',
]

CAT_COLS_CANON = [
    # fija (ajústala a tu dataset real)
    'pass_height','pass_technique','body_part','corner_side','corner_execution_time_label','game_state'
]

def preprocess_base(path: str, inference: bool) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    print(df.columns)

    df = df.drop(columns=DROP_COLS, errors='ignore')

    print(df.columns)

    if inference:
        df = df.drop(columns=EXCLUDE_COLS, errors='ignore')
    else:
        df = df.drop(columns=IDENTIFIER, errors='ignore')

    print(df.columns)
    # Consistencias/NaN
    df = df.dropna(subset=['P0_n_defenders_in_18yd_box', 'P1_n_defenders_in_18yd_box'])
    df['P1_GK_x'] = df['P1_GK_x'].fillna(-1)
    df['P1_GK_y'] = df['P1_GK_y'].fillna(-1)

    return df
