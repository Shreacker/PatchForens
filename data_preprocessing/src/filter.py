import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseFilterMethod(ABC):
    def __init__(self):
        self.score_dict = dict()

    @abstractmethod
    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray = None,
        bins: int = None
    ) -> dict[int, float]:
        ...

class OutOfCoreContingencyFilter(BaseFilterMethod):
    def _build_joint_counts(
            self,
            X: np.ndarray,
            y: np.ndarray,
            indices: np.ndarray,
            bins: int,
            batch_size=100_000
        ):
        n_cols = X.shape[1]
        n_samples = len(indices) if indices is not None else X.shape[0]
        
        # 1. Map Y to discrete integer classes
        unique_y, y_mapped = np.unique(y, return_inverse=True)
        num_y = len(unique_y)
        
        # 2. Get global quantiles for X using a memory-safe random sample
        sample_size = min(1_000_000, n_samples)
        if indices is not None:
            sample_idx = np.random.choice(indices, size=sample_size, replace=False)
        else:
            sample_idx = np.random.choice(n_samples, size=sample_size, replace=False)
        sample_idx.sort()
        
        sample_buffer = []
        for i in range(0, sample_size, batch_size):
            sample_buffer.append(X[sample_idx[i:i+batch_size]])
        X_sample = np.vstack(sample_buffer)
        
        q = np.linspace(0, 100, bins + 1)
        quantiles_raw = np.percentile(X_sample, q, axis=0)
        
        # Handle columns with duplicate percentiles (e.g. constant columns)
        quantiles_list = []
        for c in range(n_cols):
            quantiles_list.append(np.unique(quantiles_raw[:, c]))
            
        # 3. Accumulate joint counts (X_binned, Y)
        num_x_bins = bins + 2 
        joint_counts = np.zeros((n_cols, num_x_bins, num_y), dtype=np.int64)
        loop_indices = indices if indices is not None else np.arange(n_samples)
        
        # Slice the array via rows
        for i in tqdm(range(0, n_samples, batch_size), desc='Reading Chunks', leave=False):
            batch_idx = loop_indices[i:i+batch_size]
            X_chunk = X[batch_idx]
            y_chunk_mapped = y_mapped[i:i+batch_size]
            
            for c in range(n_cols):
                # Bin the floats into discrete integers directly in RAM
                x_binned = np.digitize(X_chunk[:, c], quantiles_list[c])
                
                # Fast 2D frequency accumulation
                flat_indices = x_binned * num_y + y_chunk_mapped
                counts = np.bincount(flat_indices, minlength=num_x_bins * num_y)
                joint_counts[c] += counts.reshape(num_x_bins, num_y)
                
        return joint_counts

    def _compute_entropies(self, joint_counts):
        eps = 1e-9
        
        total = np.sum(joint_counts, axis=(1, 2), keepdims=True)
        P_xy = joint_counts / total
        
        P_x = np.sum(P_xy, axis=2) 
        P_y = np.sum(P_xy, axis=1) 
        
        H_x = -np.sum(P_x * np.log2(P_x + eps), axis=1)
        H_y = -np.sum(P_y * np.log2(P_y + eps), axis=1)
        H_xy = -np.sum(P_xy * np.log2(P_xy + eps), axis=(1, 2))
        
        return H_x, H_y, H_xy

class InformationGain(OutOfCoreContingencyFilter):
    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            indices: np.ndarray = None,
            bins: int = 10
        ):
        joint_counts = self._build_joint_counts(X, y, indices, bins)
        H_x, H_y, H_xy = self._compute_entropies(joint_counts)
        
        IG = H_x + H_y - H_xy
        
        for c in range(X.shape[1]):
            self.score_dict[c] = IG[c]
            
        return dict(sorted(self.score_dict.items(), key=lambda item: item[1], reverse=True))

class SymmetricUncertainty(OutOfCoreContingencyFilter):
    def score(
            self,
            X: np.ndarray,
            y: np.ndarray,
            indices: np.ndarray = None,
            bins: int = 10
        ):
        joint_counts = self._build_joint_counts(X, y, indices, bins)
        H_x, H_y, H_xy = self._compute_entropies(joint_counts)
        
        IG = H_x + H_y - H_xy
        SU = 2 * IG / (H_x + H_y + 1e-9)
        
        for c in range(X.shape[1]):
            self.score_dict[c] = SU[c]
            
        return dict(sorted(self.score_dict.items(), key=lambda item: item[1], reverse=True))

class BaseScoreCombiner(ABC):
    @abstractmethod
    def combine(self, scores: list[dict[int, float]]) -> list[int]:
        ...

    def _cut(self, scores: dict[int, float]) -> list[int]:
        if getattr(self, 'top_k', None):
            return sorted(scores, key=scores.get, reverse=True)[:self.top_k]
        
        if getattr(self, 'threshold', None):
            return [k for k, v in scores.items() if v >= self.threshold]
        
        raise ValueError('Specify either top_k or threshold.')

class MeanCombiner(BaseScoreCombiner):
    def __init__(
            self,
            top_k: int = 1,
            threshold: float = None
        ):
        self.top_k = top_k
        self.threshold = threshold
    
    def combine(
            self,
            scores: list[dict[int, float]]
        ):
        keys = scores[0].keys()
        merged = {k: np.mean([s[k] for s in scores]) for k in keys}
        
        return self._cut(merged)
    
class IntersectCombiner(BaseScoreCombiner):
    def __init__(
            self,
            top_k: int = 1,
            min_agreement: int = 1
        ):
        self.top_k = top_k
        self.min_agreement = min_agreement

    def combine(
            self,
            scores: list[dict[int, float]]
        ):
        min_agree = self.min_agreement or len(scores)

        def top_keys(s):
            return set(sorted(s, key=s.get, reverse=True)[:self.top_k])
        
        agreement = {
            k: sum(k in top_keys(s) for s in scores)
            for k in scores[0].keys()
        }
        
        return [k for k, count in agreement.items() if count >= min_agree]
    
class Filter:
    def __init__(
            self,
            methods: list[BaseFilterMethod],
            combiner: BaseScoreCombiner = None
        ):
        self.methods = methods
        self.combiner = combiner
        self.selected_columns = None
    
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            indices: np.ndarray = None,
            n_bins: int = 10
        ):
        print('Start calculating feature scores...')
        
        scores = []
        for i, m in enumerate(self.methods):
            print(f"\n-> Running Filter Method {i+1}/{len(self.methods)}: {m.__class__.__name__}")
            scores.append(m.score(X, y, indices, bins=n_bins))
            
        print('\nChoosing columns to keep...')
        self.selected_columns = (
            list(scores[0].keys())
            if len(scores) == 1 and not self.combiner
            else self.combiner.combine(scores)
        )
        return self
    
    def transform(
            self,
            X: np.ndarray
        ):
        if self.selected_columns is None:
            raise RuntimeError("Call fit before transform")
        
        # NOTE: If X is a large memmap in size, this will return a new array in RAM.
        # Alternatively, just use the `filter.selected_columns` array!
        return X[:, self.selected_columns]
    
    def fit_transform(
            self,
            X: np.ndarray,
            y: np.ndarray,
            indices: np.ndarray = None
        ):
        self.fit(X, y, indices)
        
        return self.transform(X)

class QuantileBoundaryFilter:
    def __init__(
            self,
            lower: float = 0.01,
            upper: float = 0.99
        ):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(
            self,
            X: np.ndarray,
            indices: np.ndarray = None,
            batch_size=100_000
        ):
        n_samples = X.shape[0] if indices is None else len(indices)
        sample_size = min(1_000_000, n_samples)
        
        if indices is None:
            idx = np.random.choice(n_samples, size=sample_size, replace=False)
        else:
            idx = np.random.choice(indices, size=sample_size, replace=False)
            
        idx.sort()
        
        sample_buffer = []
        for i in range(0, sample_size, batch_size):
            batch_idx = idx[i:i+batch_size]
            sample_buffer.append(X[batch_idx])
            
        X_sample = np.vstack(sample_buffer)
        
        self.lower_bounds_ = np.percentile(X_sample, self.lower * 100, axis=0)
        self.upper_bounds_ = np.percentile(X_sample, self.upper * 100, axis=0)
        
        return self
    
    def transform(
            self,
            X: np.ndarray,
            indices: np.ndarray = None,
            batch_size=100_000
        ):
        if self.lower_bounds_ is None:
            raise RuntimeError('Call fit() before transform().')
            
        n_samples = X.shape[0] if indices is None else len(indices)
        valid_mask = np.ones(n_samples, dtype=bool)
        
        for i in range(0, n_samples, batch_size):
            if indices is None:
                X_batch = X[i:i+batch_size]
            else:
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                
            within_bounds = (X_batch >= self.lower_bounds_) & (X_batch <= self.upper_bounds_)
            valid_mask[i:i+batch_size] = within_bounds.all(axis=1)
            
        if indices is None:
            return np.where(valid_mask)[0]
        else:
            return indices[valid_mask]