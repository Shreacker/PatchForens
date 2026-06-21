import numpy as np
import pandas as pd
import gc
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA

class SBSS:
    def __init__(
            self,
            feat_matrix: np.ndarray,
            meta_df: pd.DataFrame,
            pca_components: int = 50,
            random_state: int = 0,
            image_id: str = 'image_id',
            label_col: str = 'label'
    ):
        self.feat_matrix = feat_matrix
        self.meta_df = meta_df
        self.pca_components = pca_components
        self.random_state = random_state
        self.image_id = image_id
        self.label = label_col

    def image_fingerprint(self):
        scaler = StandardScaler()
        pca = IncrementalPCA(n_components=self.pca_components)
        
        batch_size = 100_000
        n_samples = self.feat_matrix.shape[0]
        
        print("Fitting Scaler incrementally...")
        for i in range(0, n_samples, batch_size):
            X_batch = self.feat_matrix[i:i+batch_size]
            scaler.partial_fit(X_batch)
            
        print("Fitting PCA incrementally...")
        for i in range(0, n_samples, batch_size):
            X_batch = self.feat_matrix[i:i+batch_size]
            X_scaled = scaler.transform(X_batch)
            pca.partial_fit(X_scaled)
            
        print("Extracting PCA features and aggregating per image...")
        image_ids = self.meta_df[self.image_id].values
        unique_images, inverse = np.unique(image_ids, return_inverse=True)
        num_images = len(unique_images)
        
        pca_sums = np.zeros((num_images, self.pca_components), dtype=np.float32)
        counts = np.bincount(inverse).astype(np.float32)
        
        for i in range(0, n_samples, batch_size):
            X_batch = self.feat_matrix[i:i+batch_size]
            X_scaled = scaler.transform(X_batch)
            X_pca = pca.transform(X_scaled)
            
            batch_inverse = inverse[i:i+batch_size]

            for j in range(self.pca_components):
                pca_sums[:, j] += np.bincount(batch_inverse, weights=X_pca[:, j], minlength=num_images)
                
        F_pca = pca_sums / counts[:, None]
        
        df_cluster = pd.DataFrame(F_pca, columns=[f'feat_{x}' for x in range(self.pca_components)])
        df_cluster[self.image_id] = unique_images
        
        C = self.meta_df.groupby(self.image_id).size().rename('Count')
        D = self.meta_df.groupby(self.image_id)[self.label].mean().rename('Density')
        
        df_stats = pd.concat([C, D], axis=1).reset_index()
        df_cluster = df_cluster.merge(df_stats, on=self.image_id)
        
        return df_cluster
    
    def kmeans(
            self,
            n_clusters: int = 5,
            plot: bool = True,
    ):
        df_cluster = self.image_fingerprint()
        X_cluster = df_cluster.drop(columns=[self.image_id]).to_numpy()
        
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = km.fit_predict(X_cluster)

        if plot:
            plot_pca = PCA(n_components=2)
            X_plot = plot_pca.fit_transform(X_cluster)
            plt.scatter(X_plot[:, 0], X_plot[:, 1], c=clusters)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title('K-Means Clustering', size=27)
            plt.tight_layout()
            plt.show()

        return clusters, df_cluster
    
    def _train_val_test_split(
            self,
            val_frac: float,
            test_frac: float,
    ):
        clusters, df_cluster = self.kmeans(plot=False)

        df_cluster['cluster_id'] = clusters
        df_cluster['split'] = 'train'
        
        for i in range(df_cluster['cluster_id'].nunique()):
            cluster_df = df_cluster[df_cluster['cluster_id'] == i]
            
            n_total = len(cluster_df)
            n_val = int(round(n_total * val_frac))
            n_test = int(round(n_total * test_frac))
            
            val_idx = cluster_df.sample(n=n_val, random_state=self.random_state).index
            df_cluster.loc[val_idx, 'split'] = 'val'
            
            remaining_df = cluster_df.drop(val_idx)
            test_idx = remaining_df.sample(n=n_test, random_state=self.random_state).index
            df_cluster.loc[test_idx, 'split'] = 'test'

        train_image_ids = df_cluster[self.image_id][df_cluster['split'] == 'train'].values
        val_image_ids = df_cluster[self.image_id][df_cluster['split'] == 'val'].values
        test_image_ids = df_cluster[self.image_id][df_cluster['split'] == 'test'].values

        train_mask = self.meta_df[self.image_id].isin(train_image_ids).values
        val_mask = self.meta_df[self.image_id].isin(val_image_ids).values
        test_mask = self.meta_df[self.image_id].isin(test_image_ids).values
        
        train_row_indices = np.where(train_mask)[0]
        val_row_indices = np.where(val_mask)[0]
        test_row_indices = np.where(test_mask)[0]
        
        y_train = self.meta_df[self.label].values[train_row_indices]
        y_val = self.meta_df[self.label].values[val_row_indices]
        y_test = self.meta_df[self.label].values[test_row_indices]

        del df_cluster, cluster_df, remaining_df 
        del train_mask, val_mask, test_mask
        gc.collect()

        return train_row_indices, val_row_indices, test_row_indices, y_train, y_val, y_test
    
    def train_val_test_split(
            self,
            val_frac: float = 0.1,
            test_frac: float = 0.1,
    ):
        assert 0.0 <= val_frac < 1.0, "val_frac must be in range [0, 1)."
        assert 0.0 <= test_frac < 1.0, "test_frac must be in range [0, 1)."
        assert val_frac + test_frac < 1.0, "val_frac + test_frac must be less than 1.0 to leave room for training data."

        return self._train_val_test_split(
            val_frac=val_frac,
            test_frac=test_frac
        )