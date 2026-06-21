import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def export_split_to_disk(
    X_source: np.memmap, 
    meta_df: pd.DataFrame, 
    row_indices: np.ndarray, 
    selected_cols: list, 
    scaler, 
    out_x_path: str,
    out_y_path: str,
    batch_size: int = 100_000
):
    n_samples = len(row_indices)
    n_features = len(selected_cols)
    
    '''
    Exporting X to memmap data
    '''
    print(f"Exporting {n_samples} rows to {out_x_path}...")
    
    X_out = np.memmap(out_x_path, dtype=np.float32, mode='w+', shape=(n_samples, n_features))
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Writing X chunks", leave=False):
        start = i
        end = min(i + batch_size, n_samples)
        batch_indices = row_indices[start:end]
        
        X_chunk = X_source[batch_indices][:, selected_cols]
        
        if scaler is not None:
            X_chunk = scaler.transform(X_chunk)
        
        X_out[start:end] = X_chunk
        
    X_out.flush()
    del X_out 
    
    '''
    Exporting y to parquet data
    '''
    print(f"Exporting metadata to {out_y_path}...")

    meta_split = meta_df.iloc[row_indices].copy()
    meta_split.to_parquet(out_y_path, index=False)
    
    print("Done!\n")