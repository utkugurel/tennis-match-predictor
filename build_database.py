import pandas as pd
import glob
import sqlite3
import numpy as np
import sys

def build_database():
    """
    Reads all atp_matches_YYYY.csv files, aggregates player stats by
    year and surface, and saves them to a SQLite database.
    """
    print("Starting database build...")
    
    # 1. Find and load all match CSVs
    # Use a broad pattern to get all years
    csv_files = glob.glob('data/atp_matches_*.csv') 
    if not csv_files:
        print("Error: No CSV files found. Make sure the 'data' directory exists and contains the CSVs.", file=sys.stderr)
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}. Error: {e}", file=sys.stderr)
            
    if not all_dfs:
        print("Error: No CSV files were successfully read.", file=sys.stderr)
        return
        
    all_matches_df = pd.concat(all_dfs, ignore_index=True)
    
    # 2. Extract match year and clean data
    # Add errors='coerce' to gracefully handle bad dates like "300"
    all_matches_df['year'] = pd.to_datetime(all_matches_df['tourney_date'], format='%Y%m%d', errors='coerce').dt.year

    # Normalize surface labels to avoid mismatch ("Clay ", "clay", etc.)
    if 'surface' in all_matches_df.columns:
        all_matches_df['surface'] = all_matches_df['surface'].astype(str).str.strip().str.title()
    
    # Ensure all stat columns exist so vintage datasets without certain measurements
    # still yield player-season rows (we'll treat missing stats as zero later).
    stat_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
    for col in stat_cols:
        for prefix in ['w_', 'l_']:
            full_col = f'{prefix}{col}'
            if full_col not in all_matches_df.columns:
                all_matches_df[full_col] = np.nan

    info_cols = ['hand', 'ht', 'age', 'rank', 'rank_points']
    for col in info_cols:
        for prefix in ['winner_', 'loser_']:
            full_col = f'{prefix}{col}'
            if full_col not in all_matches_df.columns:
                all_matches_df[full_col] = np.nan

    # Drop rows where we don't have the essential identifiers
    all_matches_df = all_matches_df.dropna(subset=['surface', 'winner_name', 'loser_name', 'year'])
    
    # 3. Define mappings for all stats
    # We capture stats for the player ('p_') and their opponent ('o_')
    
    winner_cols = {'year': 'year', 'surface': 'surface', 'winner_id': 'player_id', 'winner_name': 'player_name'}
    loser_cols = {'year': 'year', 'surface': 'surface', 'loser_id': 'player_id', 'loser_name': 'player_name'}
    
    # Player stats
    for col in stat_cols:
        winner_cols[f'w_{col}'] = f'p_{col}'
        loser_cols[f'l_{col}'] = f'p_{col}'
        
    # Opponent stats
    for col in stat_cols:
        winner_cols[f'l_{col}'] = f'o_{col}'
        loser_cols[f'w_{col}'] = f'o_{col}'
        
    # Info stats to average or take last
    info_cols = ['hand', 'ht', 'age', 'rank', 'rank_points']
    for col in info_cols:
        winner_cols[f'winner_{col}'] = f'p_{col}'
        loser_cols[f'loser_{col}'] = f'p_{col}'
        
    winners_df = all_matches_df[winner_cols.keys()].rename(columns=winner_cols)
    losers_df = all_matches_df[loser_cols.keys()].rename(columns=loser_cols)
    
    winners_df['wins'] = 1
    winners_df['losses'] = 0
    losers_df['wins'] = 0
    losers_df['losses'] = 1
    
    # 4. Combine into one unified dataset
    full_stats_df = pd.concat([winners_df, losers_df], ignore_index=True)
    
    # 5. Group by player, year, and surface to get season-level totals
    
    # Define aggregation logic
    agg_logic = {}
    
    # Sum all match stats
    p_cols_to_sum = [f'p_{col}' for col in stat_cols]
    o_cols_to_sum = [f'o_{col}' for col in stat_cols]
    agg_logic.update({col: 'sum' for col in p_cols_to_sum})
    agg_logic.update({col: 'sum' for col in o_cols_to_sum})
    agg_logic.update({'wins': 'sum', 'losses': 'sum'})
    
    # Average info stats
    info_cols_to_avg = ['p_ht', 'p_age', 'p_rank', 'p_rank_points']
    agg_logic.update({col: 'mean' for col in info_cols_to_avg})
    
    # Take the last known hand (it shouldn't change)
    agg_logic['p_hand'] = 'last'

    # Fill NaNs with 0 for summing, but not for averaging
    sum_cols = list(agg_logic.keys())
    sum_cols.remove('p_hand') # 'last' doesn't need fillna(0)
    for col in info_cols_to_avg:
        sum_cols.remove(col) # 'mean' shouldn't have NaNs filled with 0
        
    full_stats_df[sum_cols] = full_stats_df[sum_cols].fillna(0)

    print("Aggregating stats... (this may take a minute)")
    aggregated_df = full_stats_df.groupby(['player_name', 'player_id', 'year', 'surface']).agg(agg_logic).reset_index()
    
    aggregated_df['matches_played'] = aggregated_df['wins'] + aggregated_df['losses']
    
    # Clean up averaged values
    for col in info_cols_to_avg:
        aggregated_df[col] = aggregated_df[col].round(2)

    # 6. Save to SQLite database
    print("Saving to database...")
    conn = sqlite3.connect('atp_stats.db')
    aggregated_df.to_sql('player_yearly_stats', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"Database 'atp_stats.db' built successfully with {len(aggregated_df)} player-seasons.")

if __name__ == "__main__":
    build_database()
