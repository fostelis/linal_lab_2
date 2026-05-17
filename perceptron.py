import numpy as np


class SingleLayerPerceptron:
    def __init__(
        self,
        n_features,
        init_type="small_random",
        loss_type="bce",
        l2_lambda=0.0,
        random_state=42,
    ):
        self.n_features = n_features
        self.init_type = init_type
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.w = self._init_weights()
        self.b = 0.0

        self.train_losses = []
        self.test_losses = []

    def _init_weights(self):
        if self.init_type == "zeros":
            return np.zeros(self.n_features)

        if self.init_type == "small_random":
            return self.rng.normal(loc=0.0, scale=0.01, size=self.n_features)

        if self.init_type == "large_random":
            return self.rng.normal(loc=0.0, scale=10.0, size=self.n_features)

        raise ValueError("Неизвестный тип инициализации")

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward_linear(self, X):
        return X @ self.w + self.b

    def forward(self, X):
        return self.sigmoid(self.forward_linear(X))

    def compute_bce_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        loss += 0.5 * self.l2_lambda * np.sum(self.w ** 2)
        return loss

    def compute_hinge_loss(self, y_true_pm, scores):
        margins = 1 - y_true_pm * scores
        loss = np.mean(np.maximum(0, margins))
        loss += 0.5 * self.l2_lambda * np.sum(self.w ** 2)
        return loss

    def compute_loss(self, y_true, X):
        if self.loss_type == "bce":
            y_pred = self.forward(X)
            return self.compute_bce_loss(y_true, y_pred)

        if self.loss_type == "hinge":
            y_true_pm = 2 * y_true - 1
            scores = self.forward_linear(X)
            return self.compute_hinge_loss(y_true_pm, scores)

        raise ValueError("Неизвестная функция потерь")

    def _bce_gradients(self, X_batch, y_batch):
        y_pred = self.forward(X_batch)
        error = y_pred - y_batch

        dw = (X_batch.T @ error) / len(X_batch)
        db = np.mean(error)

        dw += self.l2_lambda * self.w

        return dw, db

    def _hinge_gradients(self, X_batch, y_batch):
        y_batch_pm = 2 * y_batch - 1
        scores = self.forward_linear(X_batch)
        margins = 1 - y_batch_pm * scores
        active = margins > 0

        if np.any(active):
            dw = -np.mean(X_batch[active] * y_batch_pm[active, None], axis=0)
            db = -np.mean(y_batch_pm[active])
        else:
            dw = np.zeros_like(self.w)
            db = 0.0

        dw += self.l2_lambda * self.w

        return dw, db

    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        lr=0.1,
        batch_size=32,
    ):
        n_samples = X_train.shape[0]

        self.train_losses = []
        self.test_losses = []

        for _ in range(epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if self.loss_type == "bce":
                    dw, db = self._bce_gradients(X_batch, y_batch)
                elif self.loss_type == "hinge":
                    dw, db = self._hinge_gradients(X_batch, y_batch)
                else:
                    raise ValueError("Неизвестная функция потерь")

                self.w -= lr * dw
                self.b -= lr * db

            train_loss = self.compute_loss(y_train, X_train)
            test_loss = self.compute_loss(y_test, X_test)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

        return self

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        if self.loss_type == "hinge":
            return (self.forward_linear(X) >= 0).astype(int)

        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
