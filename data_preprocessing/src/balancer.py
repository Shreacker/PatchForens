import sys
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseBalancer(ABC):
    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        ...

class ImageUndersamplingBinary(BaseBalancer):
    def fit_transform(
        self,
        indices: np.ndarray,
        meta_df: pd.DataFrame,
        ratio: float = 1.0,
        keep_original_ratio: float = 0.05,
        image_id_col: str = 'image_id',
        label_col: str = 'label',
        random_state: int = 42
    ) -> np.ndarray:
        
        print("Balancing dataset indices...")
        
        df = meta_df.iloc[indices].copy()
        
        kept_indices = []
        
        count_1 = df.groupby(image_id_col)[label_col].sum()
        count_0 = df.groupby(image_id_col)[label_col].count() - count_1
        
        manipulated_images = count_1[count_1 > 0].index
        original_images = count_1[count_1 == 0].index
        
        # ==========================================
        # A. Process Manipulated Images
        # ==========================================
        df_manipulated = df[df[image_id_col].isin(manipulated_images)]
        
        # Keep ALL label=1 patches
        df_1 = df_manipulated[df_manipulated[label_col] == 1]
        kept_indices.extend(df_1.index.values)
        
        # Undersample label=0 patches
        df_0_manipulated = df_manipulated[df_manipulated[label_col] == 0]
        
        target_0_counts = (count_1.loc[manipulated_images] * ratio).astype(int)
        target_0_counts = np.minimum(target_0_counts, count_0.loc[manipulated_images])
        
        if not df_0_manipulated.empty:
            df_0_man_shuffled = df_0_manipulated.sample(frac=1.0, random_state=random_state)
            df_0_man_shuffled['rank'] = df_0_man_shuffled.groupby(image_id_col).cumcount()
            
            allowed_counts = df_0_man_shuffled[image_id_col].map(target_0_counts)
            df_0_man_kept = df_0_man_shuffled[df_0_man_shuffled['rank'] < allowed_counts]
            
            kept_indices.extend(df_0_man_kept.index.values)
            
        # ==========================================
        # B. Process Original (Pristine) Images
        # ==========================================
        if keep_original_ratio > 0.0:
            df_original = df[df[image_id_col].isin(original_images)]
            
            if not df_original.empty:
                df_orig_shuffled = df_original.sample(frac=1.0, random_state=random_state)
                df_orig_shuffled['rank'] = df_orig_shuffled.groupby(image_id_col).cumcount()
                
                target_orig_counts = (count_0.loc[original_images] * keep_original_ratio).astype(int)
                allowed_orig_counts = df_orig_shuffled[image_id_col].map(target_orig_counts)
                
                df_orig_kept = df_orig_shuffled[df_orig_shuffled['rank'] < allowed_orig_counts]
                kept_indices.extend(df_orig_kept.index.values)

        balanced_indices = np.array(kept_indices, dtype=np.int64)
        np.random.seed(random_state)
        np.random.shuffle(balanced_indices)
        
        print(f"Original indices: {len(indices)} -> Balanced indices: {len(balanced_indices)}")
        return balanced_indices