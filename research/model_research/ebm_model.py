import pandas as pd
import numpy as np
import optuna
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

def train_ebm_pos(X_train_pos, y_train_pos, X_test_pos, y_test_pos):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_bins = trial.suggest_int("max_bins", 64, 512)
        interactions = trial.suggest_int("interactions", 0, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 100)
        max_leaves = trial.suggest_int("max_leaves", 2, 64)

        ebm = ExplainableBoostingClassifier(
            learning_rate=learning_rate,
            max_bins=max_bins,
            interactions=interactions,
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            n_jobs=-2
        )

        ebm.fit(X_train_pos, y_train_pos)
        y_pred = ebm.predict(X_test_pos)

        return f1_score(y_test_pos, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best parameters:", study.best_params)
    print("Best F1 score:", study.best_value)

    best_ebm = ExplainableBoostingClassifier(
        **study.best_params,
        n_jobs=-2
    )
    best_ebm.fit(X_train_pos, y_train_pos)

    # Save the trained model
    joblib.dump(best_ebm, 'ebm_pos.pkl')


def train_ebm_neg(X_train_neg, y_train_neg, X_test_neg):
    ebm = ExplainableBoostingClassifier(
        learning_rate=0.001,
        max_bins=512,
        interactions=100,
        min_samples_leaf=100,
        early_stopping_rounds=50,
        n_jobs=-2, 
        random_state=42
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_neg)
    ebm.fit(X_train_neg, y_train_neg, sample_weight=sample_weights)

    # Save the trained model
    joblib.dump(ebm, 'ebm_neg.pkl')


def main():
    df1 = pd.read_csv('positive.csv', na_values=['NULL'])
    df2 = pd.read_csv('negative.csv', na_values=['NULL'])

    df = pd.concat([df1, df2], ignore_index=True)
    df['AcquisitionDateTime_DT'] = pd.to_datetime(df['AcquisitionDateTime_DT'])

    X = df.drop(columns=["PatientID", "12SL_Codes", "Phys_Codes", "TestID", "Source", 
                         "Gender", "PatientAge", "AcquisitionDateTime_DT", "MI_Phys"])
    y = df["MI_Phys"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Save the original algorithm's prediction for test set before dropping it
    y_12SL = X_test["MI_12SL"]

    # X_train = X_train.drop(columns=["MI_12SL"])
    # X_test = X_test.drop(columns=["MI_12SL"])

    # False positive
    X_train_pos = X_train[X_train["MI_12SL"] == 1].drop(columns=["MI_12SL"])
    X_test_pos = X_test[X_test["MI_12SL"] == 1].drop(columns=["MI_12SL"])
    y_train_pos = y_train.loc[X_train_pos.index]
    y_test_pos = y_test.loc[X_test_pos.index]

    # False negatives
    X_train_neg = X_train[X_train["MI_12SL"] == 0].drop(columns=["MI_12SL"])
    X_test_neg = X_test[X_test["MI_12SL"] == 0].drop(columns=["MI_12SL"])
    y_train_neg = y_train.loc[X_train_neg.index]
    y_test_neg = y_test.loc[X_test_neg.index]

    train_ebm_pos(X_train_pos, y_train_pos, X_test_pos, y_test_pos)
    train_ebm_neg(X_train_neg, y_train_neg, X_test_neg)

if __name__ == "__main__":
    main()
