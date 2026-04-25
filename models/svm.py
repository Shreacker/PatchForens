import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, epochs: int, lr: float, C: float):
        self.epochs = epochs
        self.lr = lr
        self.C = C

        self.losses = []
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        max_loss = 0
        self.w = np.zeros((d, ), dtype=np.float64)
        self.b = 0
        with tqdm(range(self.epochs), desc='Training') as pbar:
            for _ in pbar:
                # breakpoint()
                y_hat = self.decision_fn(X)
                delta = 1 - y_hat * y

                for i, delta_i in enumerate(delta):
                    if delta_i > 0:
                        w_grad = self.w + self.C * y[i] * X[i]
                        b_grad = -self.C * y[i]
                    else:
                        w_grad = self.w
                        b_grad = 0

                    self.w -= w_grad * self.lr
                    self.b -= b_grad * self.lr

                l = self.hinge_loss(y, y_hat)
                self.losses.append(l)
                if l > max_loss: max_loss = l
                perc_l = (l - max_loss) / max_loss * 100.

                pbar.set_postfix({
                    "loss change percentage": round(perc_l, 4)
                })

        return self.losses

    def decision_fn(self, X: np.ndarray):
        return self.w.T @ X.T + self.b

    def hinge_loss(self, y: np.ndarray, y_hat: np.ndarray):
        delta = 1 - y * y_hat

        return 0.5 * (self.w.T @ self.w) + \
                self.C * np.where(delta > 0, delta, 0).mean() #.sum()