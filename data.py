import numpy as np


def stratified_train_test_split(X, y, test_size=0.3, random_state=42):
    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        n_test = int(round(len(cls_indices) * test_size))

        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_fit_transform(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


def apply_standardization(X, mean, std):
    std = np.where(std == 0, 1, std)
    return (X - mean) / std


def flip_labels(y, noise=0.0, random_state=42):
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    mask = rng.random(len(y)) < noise
    y_noisy[mask] = 1 - y_noisy[mask]
    return y_noisy


def generate_linear_gaussians(
    n_samples=500,
    centers=((-2, -2), (2, 2)),
    covariance=((0.7, 0), (0, 0.7)),
    noise=0.0,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    n_class_zero = n_samples // 2
    n_class_one = n_samples - n_class_zero

    X_zero = rng.multivariate_normal(centers[0], covariance, size=n_class_zero)
    X_one = rng.multivariate_normal(centers[1], covariance, size=n_class_one)

    X_data = np.vstack([X_zero, X_one])
    y_data = np.array([0] * n_class_zero + [1] * n_class_one)

    indices = rng.permutation(n_samples)
    X_data = X_data[indices]
    y_data = y_data[indices]

    y_data = flip_labels(y_data, noise=noise, random_state=random_state + 1)

    return X_data, y_data


def generate_xor(n_samples=500, noise=0.0, random_state=42):
    rng = np.random.default_rng(random_state)

    X_data = rng.uniform(-1, 1, size=(n_samples, 2))
    y_data = (X_data[:, 0] * X_data[:, 1] > 0).astype(int)
    y_data = flip_labels(y_data, noise=noise, random_state=random_state + 1)

    return X_data, y_data


def generate_circle(n_samples=500, radius=0.7, noise=0.0, random_state=42):
    rng = np.random.default_rng(random_state)

    X_data = rng.uniform(-1.2, 1.2, size=(n_samples, 2))
    distances = np.sqrt(X_data[:, 0] ** 2 + X_data[:, 1] ** 2)
    y_data = (distances > radius).astype(int)
    y_data = flip_labels(y_data, noise=noise, random_state=random_state + 1)

    return X_data, y_data
