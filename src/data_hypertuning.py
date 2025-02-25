import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def tune_optuna_random_forest(X_train, y_train, n_trials=50):
    """
    Optimizes hyperparameters for RandomForestClassifier using Optuna.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_trials (int): Number of trials for Optuna optimization.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    # Define the objective function
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 50, step=5)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        # Create the model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            class_weight='balanced'
        )

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_auc_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            rf.fit(X_train_cv, y_train_cv)
            y_pred_prob = rf.predict_proba(X_val_cv)[:, 1]
            roc_auc_scores.append(roc_auc_score(y_val_cv, y_pred_prob))

        # Return mean ROC-AUC as the objective value
        return sum(roc_auc_scores) / len(roc_auc_scores)

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')  # Maximize ROC-AUC
    study.optimize(objective, n_trials=n_trials)

    # Return the best parameters
    return study.best_params

