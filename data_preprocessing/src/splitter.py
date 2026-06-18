import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, '../')
from ..utils.dataset import Dataset

class SBSS:
    def __init__(
            self,
            ds: Dataset,
            non_pca_cols: np.ndarray=None,
            pca_components: int=50,
            random_state: int=0,
            image_id: str='image_id',
            label_col: str='label'
    ):

        self.X, self.y = ds[:]
        self.non_pca = non_pca_cols
        self.pca_components = pca_components
        self.random_state = random_state
        self.image_id = image_id
        self.label = label_col

    def image_fingerprint(
            self
    ):
    
        X = self.X.drop(self.non_pca, axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=self.pca_components)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=[f'feat_{x}' for x in np.arange(self.pca_components)])
        for i, cols in enumerate(self.non_pca):
            df_pca.insert(i, cols, self.X[cols])
        
        feat_cols = df_pca.filter(like='feat_').columns
        C = df_pca.groupby(self.image_id).size()
        C = pd.DataFrame(C, columns=['Count'])
        D = df_pca.groupby(self.image_id)[self.label].sum() / C['Count']
        D = pd.DataFrame(D, columns=['Density'])
        F = df_pca.groupby(self.image_id)[feat_cols].mean()
        df_cluster = pd.concat([C, D, F], axis=1)
        
        return df_cluster.reset_index()
    
    def kmeans(
            self,
            n_clusters: int=5,
            plot: bool=True,
    ):

        df_cluster = self.image_fingerprint()
        X_cluster = df_cluster.drop([self.image_id], axis=1).to_numpy()
        
        # Fit K-Means Clustering Model
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = km.fit_predict(X_cluster)

        # Plotting K-Means Clustering
        if plot:
            plot_pca = PCA(n_components=2)
            X_plot = plot_pca.fit_transform(X_cluster)
            plt.scatter(X_plot[:, 0], X_plot[:, 1], c=clusters)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title('K-Means Clustering', size=27)
            plt.tight_layout()
            plt.show()

        return clusters
    
    def _train_test_split(
            self,
            test_frac: np.float64 = None,
            test_size: np.int64 = None,
    ):
        
        clusters = self.kmeans(plot=False)

        df_split = pd.DataFrame(clusters, columns=['cluster_id']).reset_index(names=['image_id'])
        df_split['train?'] = np.zeros((len(df_split)), dtype=np.int8)
        
        for i in range(df_split['cluster_id'].nunique()):
            cluster_df = df_split[df_split['cluster_id'] == i][self.image_id]
            
            n = None if test_size is None else (len(cluster_df) - test_size)
            frac = None if test_frac is None else (1 - test_frac)
            
            train_idx = cluster_df.sample(n=n, frac=frac, random_state=self.random_state)
            df_split.loc[train_idx, 'train?'] = 1

        # Train 'n Test indices split
        train_idx = df_split[self.image_id][df_split['train?'] == 1].to_numpy()
        test_idx = df_split[self.image_id][df_split['train?'] == 0].to_numpy()

        # Train 'n Test data split
        feat_cols = self.X.filter(like='feat_').columns
        cols = feat_cols.insert(0, self.image_id)
        train_data = self.X[self.X[self.image_id].isin(train_idx)]
        X_train = train_data[cols]
        y_train = train_data[self.label]
        test_data = self.X[self.X[self.image_id].isin(test_idx)]
        X_test = test_data[feat_cols]
        y_test = test_data[self.label]

        return X_train, X_test, y_train, y_test
    
    def train_test_split(
            self,
            test_frac: float | None = None,
            test_size: int | None = None,
    ):
        
        assert (test_frac > 0.0 and test_frac < 1.0), "test_frac must be in range [0, 1]."
        assert (test_size >= 0), "test_size cannot be a negative value."

        if test_size is None and test_frac is None:
            raise ValueError

        if test_size is not None:
            n = test_size
            frac = None
        
        if test_frac is not None:
            frac = test_frac
            n = None

        return self._train_test_split(
            test_frac=frac,
            test_size=n
        )