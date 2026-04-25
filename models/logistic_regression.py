import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, epoch: int, lr: float, solver='default'):
        self.epoch = epoch
        self.lr = lr
        self.solver = solver

        self.losses = []
        self.w = None
        self.b = None

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15)
        return -l.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, ), dtype=np.float64)
        self.b = np.zeros(1)

        if self.solver == 'default':
            self.w_grad = np.zeros((d,), dtype=np.float64)
            self.b_grad = np.zeros(1, dtype=np.float64)
        
        elif self.solver == 'sag' or self.solver == 'saga':
            self.w_grad = np.zeros((N, d), dtype=np.float64)
            self.b_grad = np.zeros((N,), dtype=np.float64)
        
        self.classes_ = np.unique(y)

        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar:
                y_hat = self.predict_proba(X)
                delta_y = (y_hat - y)
                self.w_G = self.w_grad.mean()
                self.b_G = self.b_grad.mean()
                
                if self.solver == 'default':
                    self.w_grad = (delta_y.T / N) @ X
                    self.w -= self.w_grad.T * self.lr
                    self.b_grad = delta_y.sum() / N
                    self.b -= self.b_grad * self.lr

                elif self.solver == 'sag':
                    for i in range(N):
                        idx = np.random.randint(N)

                        p_i = self.sigmoid(X[idx] @ self.w)
                        wg_i = (p_i - y[idx]) * X[idx]
                        bg_i = (p_i - y[idx])
                        
                        # Backward pass on weight
                        self.w_G += (wg_i - self.w_grad[idx]) / N
                        self.w -= self.lr * self.w_G
                        self.w_grad[idx] = wg_i

                        # Backward pass on bias
                        self.b_G += (bg_i - self.b_grad[idx]) / N
                        self.b -= self.lr * self.b_G
                        self.b_grad[idx] = bg_i
                
                elif self.solver == 'saga':
                    for i in range(N):
                        idx = np.random.randint(N)

                        p_i = self.sigmoid(X[idx] @ self.w)
                        wg_i = (p_i - y[idx]) * X[idx]
                        bg_i = (p_i - y[idx])

                        # Backward pass on weight
                        self.w_v = wg_i - self.w_grad[idx] + self.w_G
                        self.w -= self.lr * self.w_v
                        self.w_G += (wg_i - self.w_grad[idx]) / N
                        self.w_grad[idx] = wg_i

                        # Backward pass on bias
                        self.b_v = bg_i - self.b_grad[idx] + self.b_G
                        self.b -= self.lr * self.b_v
                        self.b_G += (bg_i - self.b_grad[idx]) / N
                        self.b_grad[idx] = bg_i

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "loss": l
                })

        return self.losses

    def sigmoid(self, X: np.ndarray):
        return np.where(
            X >= 0,
            1. / (1. + np.exp(-X)),
            np.exp(X) / (1 + np.exp(X))
        )

    def predict_proba(self, X: np.ndarray):
        y_hat = X @ self.w + self.b
        return self.sigmoid(y_hat)
    
    def predict(self, X: np.ndarray):
        is_binary = self.classes_.size <= 2
        if is_binary:
            return (self.predict_proba(X) >= 0.5).astype(int)