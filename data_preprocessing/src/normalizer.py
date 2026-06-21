import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseNormalizer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, indices: np.ndarray = None):
        ...
    
    @abstractmethod
    def partial_fit(self, X_batch: np.ndarray):
        ...
    
    @abstractmethod
    def transform(self, X: np.ndarray, out_path: str = None):
        ...

class RobustScaler(BaseNormalizer):
    def __init__(
            self,
            max_sample_size: int = 1_000_000,
            eps: float = 1e-8
        ):
        self.median = None
        self.IQR = None
        self.eps = eps
        
        self.max_sample_size = max_sample_size
        self._sample_buffer = []
        self._samples_seen = 0
    
    def partial_fit(
            self,
            X_batch: np.ndarray
        ):
        n_batch = X_batch.shape[0]
        
        if self._samples_seen < self.max_sample_size:
            take = min(self.max_sample_size - self._samples_seen, n_batch)
            idx = np.random.choice(n_batch, size=take, replace=False)
            self._sample_buffer.append(X_batch[idx].copy())
            self._samples_seen += take
            X_concat = np.vstack(self._sample_buffer)
            Q1 = np.percentile(X_concat, 25, axis=0)
            Q3 = np.percentile(X_concat, 75, axis=0)
            self.IQR = Q3 - Q1
            
            scale = np.median(np.abs(X_concat), axis=0)
            self.median = np.median(X_concat, axis=0)
            
            self.IQR = np.maximum(self.IQR, self.eps * scale)
            self.IQR[self.IQR == 0] = self.eps
            
        return self
    
    def fit(
            self,
            X: np.ndarray,
            indices: np.ndarray = None,
            batch_size: int = 100_000
        ):
        self._samples_seen = 0
        self._sample_buffer = []
        
        n_samples = len(indices) if indices is not None else X.shape[0]
        sample_size = min(self.max_sample_size, n_samples)
        
        # Calculate the percentage of rows needed to randomly sample from each batch
        fraction = sample_size / n_samples
        
        loop_indices = indices if indices is not None else np.arange(X.shape[0])
        
        # Sequential read to prevent disk thrashing
        for i in tqdm(range(0, n_samples, batch_size), desc="Fitting RobustScale", leave=False):
            batch_idx = loop_indices[i:i+batch_size]
            
            X_chunk = X[batch_idx]
            
            n_chunk = X_chunk.shape[0]
            take = int(n_chunk * fraction)
            if take > 0:
                local_idx = np.random.choice(n_chunk, size=take, replace=False)
                self._sample_buffer.append(X_chunk[local_idx].copy())
                self._samples_seen += take
                
        # Finalize medians
        X_concat = np.vstack(self._sample_buffer)
        Q1 = np.percentile(X_concat, 25, axis=0)
        Q3 = np.percentile(X_concat, 75, axis=0)
        self.IQR = Q3 - Q1
        
        scale = np.median(np.abs(X_concat), axis=0)
        self.median = np.median(X_concat, axis=0)
        
        self.IQR = np.maximum(self.IQR, self.eps * scale)
        self.IQR[self.IQR == 0] = self.eps
        
        return self
    
    def transform(
            self,
            X: np.ndarray,
            out_path: str = None,
            batch_size: int = 100_000
        ):
        if self.median is None or self.IQR is None:
            raise ValueError("RobustScale has not been fitted yet.")
            
        if out_path is not None:
            X_out = np.memmap(out_path, dtype=np.float32, mode='w+', shape=X.shape)
            n_samples = X.shape[0]
        
            for i in tqdm(range(0, n_samples, batch_size), desc="Writing scaled memmap", leave=False):
                X_out[i:i+batch_size] = (X[i:i+batch_size] - self.median) / self.IQR
            X_out.flush()
        
            return X_out
        
        else:
            return (X - self.median) / self.IQR

class StandardScaler(BaseNormalizer):
    def __init__(
            self,
            eps: float = 1e-8
        ):
        self.mean = None
        self.var = None
        self.eps = eps
        self.n_samples_seen = 0
    
    def partial_fit(self, X_batch: np.ndarray):
        if self.mean is None:
            self.mean = np.mean(X_batch, axis=0, dtype=np.float64)
            self.var = np.var(X_batch, axis=0, dtype=np.float64)
            self.n_samples_seen = X_batch.shape[0]
    
        else:
            n_a = self.n_samples_seen
            n_b = X_batch.shape[0]
            n_total = n_a + n_b
            
            mean_b = np.mean(X_batch, axis=0, dtype=np.float64)
            var_b = np.var(X_batch, axis=0, dtype=np.float64)
            
            delta = mean_b - self.mean
            self.mean = self.mean + delta * n_b / n_total
            
            m_a = self.var * n_a
            m_b = var_b * n_b
            M2 = m_a + m_b + (delta ** 2) * n_a * n_b / n_total
            self.var = M2 / n_total
            self.n_samples_seen = n_total
            
        return self
    
    def fit(
            self,
            X: np.ndarray,
            indices: np.ndarray = None,
            batch_size: int = 100_000
        ):
        self.mean = None
        self.var = None
        self.n_samples_seen = 0
        
        n_samples = len(indices) if indices is not None else X.shape[0]
        loop_indices = indices if indices is not None else np.arange(X.shape[0])
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Fitting StandardScaler", leave=False):
            batch_idx = loop_indices[i:i+batch_size]
            self.partial_fit(X[batch_idx])
            
        return self
    
    @property
    def stdv(self):
        if self.var is None: return None
        std = np.sqrt(self.var)
        std = np.maximum(std, self.eps)
        std[std == 0] = self.eps
        return std
    
    def transform(
            self,
            X: np.ndarray,
            out_path: str = None,
            batch_size: int = 100_000
        ):
        if self.mean is None or self.var is None:
            raise ValueError("StandardScaler has not been fitted yet.")
            
        if out_path is not None:
            X_out = np.memmap(out_path, dtype=np.float32, mode='w+', shape=X.shape)
            n_samples = X.shape[0]
            for i in tqdm(range(0, n_samples, batch_size), desc="Writing scaled memmap", leave=False):
                X_out[i:i+batch_size] = (X[i:i+batch_size] - self.mean) / self.stdv
            X_out.flush()
        
            return X_out
        
        else: 
            return (X - self.mean) / self.stdv