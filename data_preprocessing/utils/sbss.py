import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SBSS:
    def __init__(
            self,
            df: pd.DataFrame=None,
            non_pca_cols: np.ndarray=None,
            pca_components: int=50,
            random_state: int=0,
            image_id: str='image_id',
            label_col: str='label'
    ):

        self.df = df
        self.non_pca = non_pca_cols
        self.pca_components = pca_components
        self.random_state = random_state
        self.image_id = image_id
        self.label = label_col

    def image_fingerprint(
            self
    ):
    
        X = self.df.drop(self.non_pca, axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=self.pca_components)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=[f'feat_{x}' for x in np.arange(self.pca_components)])
        for i, cols in enumerate(self.non_pca):
            df_pca.insert(i, cols, self.df[cols])
        
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
    
    def train_test_split(
            self,
            test_size: np.float32=0.2,
            plot_cluster: bool=False,
    ):
        clusters = self.kmeans(plot=plot_cluster)

        df_split = pd.DataFrame(clusters, columns=['cluster_id']).reset_index(names=['image_id'])
        df_split['train?'] = np.zeros((len(df_split)), dtype=np.int8)
        
        for i in range(df_split['cluster_id'].nunique()):
            train_idx = df_split[df_split['cluster_id'] == i][self.image_id].sample(frac=1-test_size, random_state=self.random_state)
            df_split.loc[train_idx, 'train?'] = 1

        # Train 'n Test indices split
        train_idx = df_split[self.image_id][df_split['train?'] == 1].to_numpy()
        test_idx = df_split[self.image_id][df_split['train?'] == 0].to_numpy()

        # Train 'n Test data split
        feat_cols = self.df.filter(like='feat_').columns
        cols = feat_cols.insert(0, self.image_id)
        train_data = self.df[self.df[self.image_id].isin(train_idx)]
        X_train = train_data[cols]
        y_train = train_data[self.label]
        test_data = self.df[self.df[self.image_id].isin(test_idx)]
        X_test = test_data[feat_cols]
        y_test = test_data[self.label]

        return X_train, X_test, y_train, y_test