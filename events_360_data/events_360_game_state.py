import pandas as pd

events = pd.read_csv('/Users/arturoreza/Documents/isac_2025/events_360_data/events_df_Liga_MX_2024_2025.csv')
matches = pd.read_csv('/Users/arturoreza/Documents/isac_2025/matches_data/season_matches_2024_2025.csv')

# Data copies
events_1 = events.copy()
matches_1 = matches.copy()

# Merging event with matches data
df = events.merge(
    matches[['match_id', 'home_team', 'away_team']],
    on='match_id',
    how='left'
)

# Prefer timestamp if present; else fall back to minute+second
if 'timestamp' in df.columns and df['timestamp'].notna().any():
    # StatsBomb-like "HH:MM:SS.mmm" or "MM:SS.mmm"
    df['ts'] = pd.to_timedelta(df['timestamp'])
else:
    df['ts'] = (pd.to_timedelta(df['minute'], unit='m')
                + pd.to_timedelta(df.get('second', 0), unit='s'))

# Order events within each match
sort_keys = ['match_id', 'period', 'ts']
if 'index' in df.columns:  # your file has an 'index' sequence – great tie-breaker
    sort_keys.append('index')
df = df.sort_values(sort_keys, kind='mergesort').reset_index(drop=True)

# Per event goal deltas from the home perspective
df['home_goal_evt'] = ((df['shot_outcome'] == 'Goal') & (df['team'] == df['home_team'])).astype(int)
df['away_goal_evt'] = ((df['shot_outcome'] == 'Goal') & (df['team'] == df['away_team'])).astype(int)

df['goal_delta'] = df['home_goal_evt'] - df['away_goal_evt']  # +1 if home scores, -1 if away scores, 0 otherwise

# Game state BEFORE the current event:
df['game_state'] = df.groupby('match_id')['goal_delta'].cumsum().shift(fill_value=0)

df = df.drop(columns=['ts', 'home_goal_evt', 'away_goal_evt', 'goal_delta'])

df.to_csv('events_df_Liga_MX_2024_2025.csv')