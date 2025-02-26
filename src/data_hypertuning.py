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

    Usage:
        best_params = tuntune_optuna_random_forest(X_train_selected, y_train_balanced, n_trials=50)
        print("Best hyperparameters:", best_params)
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


from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def tune_hyperopt_random_forest(X_train, y_train, max_evals=50):
    """
    Optimizes hyperparameters for RandomForestClassifier using Hyperopt.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        max_evals (int): Number of evaluations for Hyperopt optimization.

    Returns:
        dict: Best hyperparameters found by Hyperopt.

    Usage:
        best_params = tune_hyperopt_random_forest(X_train, y_train, n_trials=50)
        print("Best hyperparameters:", best_params)
    """
    # Define the search space
    search_space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 5, 50, 5),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
    }

    # Define the objective function
    def objective(params):
        # Convert float hyperparameters to integers
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        # Create the model
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
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

        # Return the negative mean ROC-AUC (since Hyperopt minimizes)
        return {'loss': -sum(roc_auc_scores) / len(roc_auc_scores), 'status': STATUS_OK}

    # Run Hyperopt optimization
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Convert best_params to integer values
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

    return best_params

# Usage
#best_params_hyperopt = tune_hyperopt_random_forest(X_train_selected, y_train_balanced, max_evals=50)
#print("Best hyperparameters using Hyperopt:", best_params_hyperopt)
