import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def algorithmic_splitting(X, alpha=0.45, beta=0.15, gamma=0.40, n_clusters=8):
    """
    X: Input DataFrame hoặc ndarray
    alpha, beta, gamma: Tỉ lệ lấy mẫu từ các vùng Median, Quartile, Extreme
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.7) # giữ lại 80% thông tin
    X_red = pca.fit_transform(X_scaled)
    
    # 2. Labeling by clustering (Dùng MiniBatch để tránh lỗi Memory với 1.9M dòng)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_red)
    
    median_indices = []
    extreme_indices = []
    quartile_indices = []
    
    for c in range(n_clusters):
        # Lấy các điểm thuộc cụm hiện tại
        idx_in_cluster = np.where(clusters == c)[0]
        points = X_red[idx_in_cluster]
        
        # 3. Calculate mu and sigma (tính trên khoảng cách từ điểm tới tâm cụm)
        center = kmeans.cluster_centers_[c]
        dist_to_center = np.linalg.norm(points - center, axis=1)
        
        mu = np.mean(dist_to_center)
        sigma = np.std(dist_to_center)
        
        # 4. Define ranges
        median_lower, median_upper = mu - sigma, mu + sigma
        extreme_lower, extreme_upper = mu - 2*sigma, mu + 2*sigma
        
        # 5. Phân loại điểm vào các vùng
        for i, d in enumerate(dist_to_center):
            global_idx = idx_in_cluster[i]
            if median_lower <= d <= median_upper:
                median_indices.append(global_idx)
            elif extreme_lower <= d <= extreme_upper:
                extreme_indices.append(global_idx)
            else:
                quartile_indices.append(global_idx)

    # 6. Sampling into sub-datasets (Train, Test, Val)
    def split_category(indices, hyparam):
        np.random.shuffle(indices)
        n = len(indices)
        train = int(n * hyparam)
        test = int((n-train) * hyparam)
        return indices[:train], indices[train:train+test], indices[train+test:]

    m_train, m_test, m_val = split_category(median_indices, alpha)
    q_train, q_test, q_val = split_category(quartile_indices, beta)
    e_train, e_test, e_val = split_category(extreme_indices, gamma)
    
    # Kết hợp các chỉ số theo tỉ lệ alpha, beta, gamma
    train_idx = np.concatenate([m_train, q_train, e_train])
    test_idx = np.concatenate([m_test, q_test, e_test])
    val_idx = np.concatenate([m_val, q_val, e_val])
    
    return train_idx, test_idx, val_idx

# --- Cách sử dụng ---
train_idx, test_idx, val_idx = algorithmic_splitting(X_df)
X_train = X_df.iloc[train_idx]
y_train = y.iloc[train_idx]
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
X_test = X_df.iloc[test_idx]
y_test = y.iloc[test_idx]
score = model.score(X_test, y_test)
print(f'Algothmic-split Score: {score}')