import math, copy
import numpy as np
import matplotlib.pyplot as plt  # plotting
import pandas as pd


def main():
    # Read the student performance CSV into a pandas DataFrame
    file = pd.read_csv('../Data/Student_Performance.csv')

    # Target y: performance index score we want to predict
    y = file['Performance Index'].to_numpy(dtype=float)

    # Feature columns (numeric only); Extracurricular Activities is not used
    feature_names = [
        'Hours Studied',
        'Previous Scores',
        'Sleep Hours',
        'Sample Question Papers Practiced',
    ]
    # Build design matrix X with one row per student and one column per feature
    X = file[feature_names].to_numpy(dtype=float)

    # Shuffle row indices so train/test split is random but reproducible
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(y))
    split = int(0.75 * len(y))  # 75% of rows for training
    train_idx = indices[:split]
    test_idx = indices[split:]

    # Split features and targets into training and test sets
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    print(f"Training examples: {len(y_train)}, test examples: {len(y_test)}")

    # Z-score normalization: fit mean and std on training data only
    mu = np.mean(X_train_raw, axis=0)       # mean of each feature (training)
    sigma = np.std(X_train_raw, axis=0)     # std of each feature (training)
    sigma = np.where(sigma == 0, 1.0, sigma)  # avoid division by zero
    # Transform: (x - mean) / std so each feature has mean 0 and std 1
    X_train = (X_train_raw - mu) / sigma
    X_test = (X_test_raw - mu) / sigma
    X_mean_centered = X_train_raw - mu  # for visualization only

    # Plot raw vs mean-centered vs normalized (Hours Studied vs Previous Scores)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    plot_sets = [
        (X_train_raw, 'unnormalized'),
        (X_mean_centered, 'mean-centered'),
        (X_train, 'Z-score normalized'),
    ]
    for i, (data, title) in enumerate(plot_sets):
        ax[i].scatter(data[:, 0], data[:, 1], alpha=0.6)
        ax[i].set_xlabel(feature_names[0])
        ax[i].set_ylabel(feature_names[1])
        ax[i].set_title(title)
    plt.tight_layout()
    plt.suptitle('Features before and after normalization')
    plt.show()

    # Number of features determines length of weight vector w
    n_features = X_train.shape[1]
    w_init = np.zeros(n_features)  # start with zero weights
    b_init = 0.0                   # start with zero bias

    # Z-scored features allow a larger learning rate; cost flattens in a few hundred steps
    iterations = 500
    tmp_alpha = 1.0

    # Run batch gradient descent to learn w and b
    w_final, b_final, cost_history = gradientDescent(
        X_train, y_train, w_init, b_init, tmp_alpha,
        iterations, computeCost, computeGradientMulti
    )
    print(f"(w, b) found by gradient descent: w={w_final}, b={b_final:.4f}")

    # Plot how training cost decreases over iterations
    plt.figure(figsize=(8, 4))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost J')
    plt.title('Training cost vs iteration')
    plt.tight_layout()
    plt.show()

    # Predictions on training set: f(x) = X @ w + b
    y_train_pred = X_train @ w_final + b_final
    # Predictions on test set (same w, b; uses test features normalized with train mu/sigma)
    y_test_pred = X_test @ w_final + b_final

    # Scatter: predicted vs actual (training)
    plot_predicted_vs_actual(y_train, y_train_pred, 'training set')
    # Scatter: predicted vs actual (test)
    plot_predicted_vs_actual(y_test, y_test_pred, 'test set')

    # Show five example test predictions
    print("\n--- Sample test predictions ---")
    print(f"{'actual':>12} {'predicted':>12} {'error':>12}")
    for i in range(min(5, len(y_test))):
        pred = y_test_pred[i]
        err = pred - y_test[i]
        print(f"{y_test[i]:12.2f} {pred:12.2f} {err:12.2f}")


def plot_predicted_vs_actual(y_true, y_pred, label):
    """Diagonal reference line: perfect predictions lie on y = x."""
    plt.figure(figsize=(6, 6))                      # new figure for this plot
    plt.scatter(y_true, y_pred, alpha=0.7, c='orange', label='Examples')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, '--', c='blue', label='Perfect prediction')  # y=x line
    plt.xlabel('Actual performance index')
    plt.ylabel('Predicted performance index')
    plt.title(f'Predicted vs actual — {label}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def computeCost(X, y, w, b):
    """Multivariable cost J = (1 / 2m) * sum((w·x + b - y)^2)."""
    m = X.shape[0]              # m = number of training examples (rows)
    f_wb = X @ w + b              # vector of all m predictions at once
    errors = f_wb - y             # element-wise prediction minus actual
    squared = errors ** 2         # square each error
    cost = np.sum(squared)        # sum of squared errors over all examples
    totalCost = (1 / (2 * m)) * cost  # scale by 1/(2m) per cost definition
    return totalCost              # return scalar cost J


def computeGradientMulti(X, y, w, b):
    """Gradients of J with respect to w (vector) and b (scalar)."""
    m = X.shape[0]                # number of training examples
    f_wb = X @ w + b              # all m predictions
    err = f_wb - y                # error vector (length m)
    dj_dw = (1 / m) * (X.T @ err) # weight gradients: average of err * x_j over rows
    dj_db = (1 / m) * np.sum(err) # bias gradient: average error over rows
    return dj_dw, dj_db           # return (gradient for w, gradient for b)


def gradientDescent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """Batch gradient descent; same pattern as salary_progression.py."""
    w = copy.deepcopy(w_in)  # work on a copy of initial weights
    b = float(b_in)          # ensure bias is a float
    J_history = []           # list to store cost after each update (for plotting)

    for i in range(num_iters):   # repeat num_iters times
        dj_dw, dj_db = gradient_function(X, y, w, b)  # compute gradients at current w, b
        w = w - alpha * dj_dw    # move weights opposite to gradient (descent)
        b = b - alpha * dj_db    # move bias opposite to gradient

        if i < 100000:   # cap history length (same guard as salary_progression.py)
            J_history.append(cost_function(X, y, w, b))  # record cost this iteration

        if i % math.ceil(num_iters / 10) == 0:   # print progress ~10 times
            print(
                f"Iteration {i:5d}: Cost {J_history[-1]:0.2e}  "
                f"w={w}  b={b:.4f}"
            )

    return w, b, J_history     # final parameters and cost history


if __name__ == '__main__':
    main()