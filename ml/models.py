"""
EduPredict Machine Learning Models & Metrics Module
Provides high-performance, scikit-learn compatible implementations of:
- Linear Regression (Ordinary Least Squares with Ridge regularization)
- Decision Tree Regressor (Recursive variance reduction / MSE criterion)
- Random Forest Regressor (Ensemble Bagging with Random Subspace feature selection)
- Model Evaluation Metrics (MAE, MSE, RMSE, R2 Score)
"""

import numpy as np


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """Calculate Coefficient of Determination (R² Score)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(np.round(n_samples * test_size))
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    if hasattr(X, 'iloc'):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test = X[train_idx], X[test_idx]
        
    if hasattr(y, 'iloc'):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        y_train, y_test = y[train_idx], y[test_idx]
        
    return X_train, X_test, y_train, y_test


class LinearRegressionModel:
    """
    Ordinary Least Squares (OLS) Linear Regression model.
    Uses closed-form Normal Equation with small ridge regularizer for numerical stability.
    beta = (X^T * X + lambda * I)^(-1) * X^T * y
    """
    def __init__(self, fit_intercept=True, alpha=1e-5):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_design = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_design = X

        # Regularized normal equation
        XtX = X_design.T @ X_design
        reg = self.alpha * np.eye(XtX.shape[0])
        if self.fit_intercept:
            reg[0, 0] = 0.0  # Do not regularize intercept

        Xty = X_design.T @ y
        beta = np.linalg.solve(XtX + reg, Xty)

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        # Normalized absolute weights as feature importance proxy
        abs_coef = np.abs(self.coef_)
        total = np.sum(abs_coef)
        self.feature_importances_ = (abs_coef / total) if total > 0 else np.zeros_like(abs_coef)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class _Node:
    """Internal binary tree node."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, impurity_reduction=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity_reduction = impurity_reduction

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorModel:
    """
    Decision Tree Regressor using recursive Mean Squared Error (variance) minimization.
    Calculates exact feature importances based on cumulative variance reduction.
    """
    def __init__(self, max_depth=8, min_samples_split=5, min_samples_leaf=3, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_ = X.shape[1]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        importances = np.zeros(self.n_features_, dtype=np.float64)
        self.root = self._grow_tree(X, y, depth=0, importances=importances)
        
        total_imp = np.sum(importances)
        if total_imp > 0:
            self.feature_importances_ = importances / total_imp
        else:
            self.feature_importances_ = np.ones(self.n_features_, dtype=np.float64) / self.n_features_
        return self

    def _grow_tree(self, X, y, depth=0, importances=None):
        n_samples, n_feats = X.shape

        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples <= self.min_samples_leaf or
            np.var(y) < 1e-7):
            return _Node(value=float(np.mean(y)))

        # Feature subsampling
        if self.max_features is None:
            feat_idxs = np.arange(n_feats)
        elif isinstance(self.max_features, int):
            feat_idxs = np.random.choice(n_feats, min(self.max_features, n_feats), replace=False)
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_feats))
            feat_idxs = np.random.choice(n_feats, k, replace=False)
        elif self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_feats)))
            feat_idxs = np.random.choice(n_feats, k, replace=False)
        else:
            feat_idxs = np.arange(n_feats)

        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if best_feat is None or best_gain <= 0:
            return _Node(value=float(np.mean(y)))

        if importances is not None:
            # Weighted variance reduction: (N / Total_N) * gain
            importances[best_feat] += (n_samples) * best_gain

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1, importances)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1, importances)
        return _Node(best_feat, best_thresh, left, right, impurity_reduction=best_gain)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_feat, split_thresh = None, None
        current_variance = np.var(y)
        n = len(y)

        for feat in feat_idxs:
            X_column = X[:, feat]
            thresholds = np.unique(X_column)
            # Sample thresholds if too many unique values for performance
            if len(thresholds) > 45:
                thresholds = np.percentile(thresholds, np.linspace(2, 98, 45))

            for thresh in thresholds:
                left_mask = X_column <= thresh
                n_left = np.sum(left_mask)
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                var_left = np.var(y[left_mask])
                var_right = np.var(y[~left_mask])
                gain = current_variance - (n_left / n * var_left + n_right / n * var_right)

                if gain > best_gain:
                    best_gain = gain
                    split_feat = feat
                    split_thresh = thresh

        return split_feat, split_thresh, best_gain

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse_tree(x, self.root) for x in X], dtype=np.float64)

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestRegressorModel:
    """
    Random Forest Regressor.
    Constructs an ensemble of randomized decision trees via bagging and random subspace selection.
    Provides prediction intervals / confidence metrics using tree variance.
    """
    def __init__(self, n_estimators=60, max_depth=10, min_samples_split=5, 
                 min_samples_leaf=2, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.n_features_ = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, self.n_features_ = X.shape
        
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        all_importances = []

        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            
            tree_seed = rng.randint(0, 100000)
            tree = DecisionTreeRegressorModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            all_importances.append(tree.feature_importances_)

        # Average feature importances across all trees
        mean_imp = np.mean(all_importances, axis=0)
        self.feature_importances_ = mean_imp / np.sum(mean_imp)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Aggregate predictions across all ensemble trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def predict_with_confidence(self, X):
        """
        Returns the mean prediction along with prediction standard deviation
        and estimated confidence percentage (higher confidence = lower tree variance).
        """
        X = np.asarray(X, dtype=np.float64)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        means = np.mean(tree_preds, axis=0)
        stds = np.std(tree_preds, axis=0)
        
        # Confidence score: maps variance to 75% - 98% scale
        # Low standard deviation (e.g. < 2 points) indicates high tree agreement
        confidence = np.clip(100.0 - (stds * 3.5), 72.0, 98.5)
        return means, stds, confidence
