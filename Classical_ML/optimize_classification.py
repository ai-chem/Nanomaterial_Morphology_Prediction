import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_curve,
)
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from sklearn.model_selection import GridSearchCV
import pickle
import xgboost
warnings.filterwarnings("ignore")


def classify_size(shape, min_size, max_size, df):
    df["{}_s".format(shape)] = (
        (df["{}_avg".format(shape)] < min_size) & (df["{}".format(shape)] == 1)
    ).astype(int)
    df["{}_m".format(shape)] = (
        (df["{}_avg".format(shape)] <= max_size)
        & (df["{}_avg".format(shape)] >= min_size)
        & (df["{}".format(shape)] == 1)
    ).astype(int)
    df["{}_l".format(shape)] = (
        (df["{}_avg".format(shape)] > max_size) & (df["{}".format(shape)] == 1)
    ).astype(int)


def create_dataset(drop_nonimportant_columns=True):
    df = pd.read_excel("../Datasets/dataset_labeled.xlsx")
    df["Cube_avg"] = df.loc[:, ["Cube_min", "Cube_max"]].mean(axis=1)
    df["Stick_avg"] = df.loc[:, ["Stick_min", "Stick_max"]].mean(axis=1)
    df["Sphere_avg"] = df.loc[:, ["Sphere_min", "Sphere_max"]].mean(axis=1)
    df["Flat_avg"] = df.loc[:, ["Flat_min", "Flat_max"]].mean(axis=1)
    df["Amorphous_avg"] = df.loc[:, ["Amorphous_min", "Amorphous_max"]].mean(axis=1)
    df = df.drop(
        columns=[
            "Image_id",
            "Cube_min",
            "Cube_max",
            "Stick_min",
            "Stick_max",
            "Sphere_min",
            "Sphere_max",
            "Flat_min",
            "Flat_max",
            "Amorphous_min",
            "Amorphous_max",
        ]
    )
    if drop_nonimportant_columns:
        df = df.drop(
            columns=[
                "Stirring, rpm",
                "Ca ion, mM",
                "CO3 ion, mM",
                "Hexadecyltrimethylammonium bromide",
                "Triton X-100",
                "1-Hexanol",
                "Methyl alcohol",
            ]
        )

    classify_size("Cube", 15, 20, df)
    classify_size("Sphere", 10, 14, df)
    classify_size("Stick", 35, 45, df)
    classify_size("Flat", 23, 30, df)
    classify_size("Amorphous", 12, 20, df)
    df = df.drop(
        columns=["Cube_avg", "Stick_avg", "Sphere_avg", "Flat_avg", "Amorphous_avg"]
    )
    return df


def calculate_metrics(model, X_test, y_test):
    y_predict_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds2 = precision_recall_curve(y_test, y_predict_proba)
    threshold = thresholds2[np.argmin(abs(precision - recall))]

    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    return acc, f1


def grid_search_rf(df, particle, rs, scaler):
    y = df[particle].to_list()
    #X = df.iloc[:, 0:29]
    X = df.iloc[:, 0:22] 
    num_cols = X.iloc[:, 0:10].columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=rs, shuffle=True
    )

    rf_def = RandomForestClassifier(random_state=rs)
    rf_def.fit(X_train, y_train)

    rf_def_acc, rf_def_f1 = calculate_metrics(rf_def, X_test, y_test)

    param_grid = {
        "n_estimators": [25, 50, 100, 150, 200, 500],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [3, 6, 9],
        "max_leaf_nodes": [3, 6, 9],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=rs)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, n_jobs=4, verbose=2, scoring="f1"
    )
    grid_search.fit(X_train, y_train)
    best_est = grid_search.best_estimator_

    rf_opt_acc, rf_opt_f1 = calculate_metrics(best_est, X_test, y_test)

    return rf_def_acc, rf_def_f1, rf_opt_acc, rf_opt_f1, best_est


def calculate_different_rs_results(particle, random_states):
    df = create_dataset()

    scaler = MinMaxScaler()

    rf_accuracies = []
    rf_f1s = []
    rf_accuracies_opt = []
    rf_f1s_opt = []

    for rs in random_states:
        rf_acc, rf_f1, rf_acc_optimized, rf_f1_optimized, best_est = grid_search_rf(
            df, particle, rs, scaler
        )
        with open(f'rf_results_filtered_15testsize/best_rf_filtered_15testsize_{particle}_{rs}.pickle', 'wb') as f:
            pickle.dump(best_est, f)
        rf_accuracies.append(rf_acc)
        rf_f1s.append(rf_f1)
        rf_accuracies_opt.append(rf_acc_optimized)
        rf_f1s_opt.append(rf_f1_optimized)

    return rf_accuracies, rf_f1s, rf_accuracies_opt, rf_f1s_opt


def calculate_final_metrics(particle):
    random_states = [0, 10, 20, 30, 40]
    (
        rf_accuracies,
        rf_f1s,
        rf_accuracies_opt,
        rf_f1s_opt,
    ) = calculate_different_rs_results(particle, random_states)
    mean_rf_def_acc = np.mean(rf_accuracies)
    std_rf_def_acc = np.std(rf_accuracies)
    mean_rf_def_f1 = np.mean(rf_f1s)
    std_rf_def_f1 = np.std(rf_f1s)

    mean_rf_opt_acc = np.mean(rf_accuracies_opt)
    std_rf_opt_acc = np.std(rf_accuracies_opt)
    mean_rf_opt_f1 = np.mean(rf_f1s_opt)
    std_rf_opt_f1 = np.std(rf_f1s_opt)

    dict = {'rs': random_states,'rf_acc_unoptimized': rf_accuracies, 'rf_f1_unoptimized': rf_f1s, 'rf_acc_optimized': rf_accuracies_opt, 'rf_f1_optimized': rf_f1s_opt} 
    df = pd.DataFrame(dict)
    df.to_csv(f'rf_results_filtered_15testsize/rf_filtered_15testsize_{particle}.csv')

    results = pd.DataFrame(
        columns=[
            "rf_acc_unoptimized",
            "rf_f1_unoptimized",
            "rf_acc_optimized",
            "rf_f1_optimized",
        ]
    )
    results.loc[len(results)] = [
        "%.2f" % mean_rf_def_acc + " ± " + "%.2f" % std_rf_def_acc,
        "%.2f" % mean_rf_def_f1 + " ± " + "%.2f" % std_rf_def_f1,
        "%.2f" % mean_rf_opt_acc + " ± " + "%.2f" % std_rf_opt_acc,
        "%.2f" % mean_rf_opt_f1 + " ± " + "%.2f" % std_rf_opt_f1,
    ]
    return results



def grid_search_xgb(df, particle, rs, scaler):
    y = df[particle].to_list()
    #X = df.iloc[:, 0:29]
    X = df.iloc[:, 0:22] 

    num_cols = X.iloc[:, 0:10].columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=rs, shuffle=True
    )

    xgb_def = xgboost.XGBClassifier(random_state=rs)
    xgb_def.fit(X_train, y_train)

    xgb_def_acc, xgb_def_f1 = calculate_metrics(xgb_def, X_test, y_test)

    param_grid = {
        "gamma": [0.5, 1, 1.5, 2, 5],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5],
        'n_estimators': [25, 50, 100, 150, 200, 500],
        'learning_rate': [0.2, 0.1, 0.01, 0.05]
    }

    xgb = xgboost.XGBClassifier(random_state=rs)
    grid_search = GridSearchCV(
        estimator=xgb, param_grid=param_grid, cv=5, n_jobs=4, verbose=2, scoring="f1"
    )
    grid_search.fit(X_train, y_train)
    best_est = grid_search.best_estimator_

    xgb_opt_acc, xgb_opt_f1 = calculate_metrics(best_est, X_test, y_test)

    return xgb_def_acc, xgb_def_f1, xgb_opt_acc, xgb_opt_f1, best_est


def calculate_different_rs_results_xgb(particle, random_states):
    df = create_dataset()

    scaler = MinMaxScaler()

    xgb_accuracies = []
    xgb_f1s = []
    xgb_accuracies_opt = []
    xgb_f1s_opt = []

    for rs in random_states:
        xgb_acc, xgb_f1, xgb_acc_optimized, xgb_f1_optimized, best_est = grid_search_xgb(
            df, particle, rs, scaler
        )
        with open(f'xgb_results_filtered_15testsize/best_xgb_filtered_15testsize_{particle}_{rs}.pickle', 'wb') as f:
            pickle.dump(best_est, f)
        xgb_accuracies.append(xgb_acc)
        xgb_f1s.append(xgb_f1)
        xgb_accuracies_opt.append(xgb_acc_optimized)
        xgb_f1s_opt.append(xgb_f1_optimized)

    return xgb_accuracies, xgb_f1s, xgb_accuracies_opt, xgb_f1s_opt


def calculate_final_metrics_xgb(particle):
    random_states = [0, 10, 20, 30, 40]
    (
        xgb_accuracies,
        xgb_f1s,
        xgb_accuracies_opt,
        xgb_f1s_opt,
    ) = calculate_different_rs_results_xgb(particle, random_states)
    mean_xgb_def_acc = np.mean(xgb_accuracies)
    std_xgb_def_acc = np.std(xgb_accuracies)
    mean_xgb_def_f1 = np.mean(xgb_f1s)
    std_xgb_def_f1 = np.std(xgb_f1s)

    mean_xgb_opt_acc = np.mean(xgb_accuracies_opt)
    std_xgb_opt_acc = np.std(xgb_accuracies_opt)
    mean_xgb_opt_f1 = np.mean(xgb_f1s_opt)
    std_xgb_opt_f1 = np.std(xgb_f1s_opt)

    dict = {'rs': random_states,'xgb_acc_unoptimized': xgb_accuracies, 'xgb_f1_unoptimized': xgb_f1s, 'xgb_acc_optimized': xgb_accuracies_opt, 'xgb_f1_optimized': xgb_f1s_opt} 
    df = pd.DataFrame(dict)
    df.to_csv(f'xgb_results_filtered_15testsize/xgb_filtered_15testsize_{particle}.csv')

    results = pd.DataFrame(
        columns=[
            "xgb_acc_unoptimized",
            "xgb_f1_unoptimized",
            "xgb_acc_optimized",
            "xgb_f1_optimized",
        ]
    )
    results.loc[len(results)] = [
        "%.2f" % mean_xgb_def_acc + " ± " + "%.2f" % std_xgb_def_acc,
        "%.2f" % mean_xgb_def_f1 + " ± " + "%.2f" % std_xgb_def_f1,
        "%.2f" % mean_xgb_opt_acc + " ± " + "%.2f" % std_xgb_opt_acc,
        "%.2f" % mean_xgb_opt_f1 + " ± " + "%.2f" % std_xgb_opt_f1,
    ]
    return results