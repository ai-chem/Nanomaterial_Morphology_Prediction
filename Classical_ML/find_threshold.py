import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_threshold(particle, df, X, rs=0):
    y = df[particle].to_list()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=rs, shuffle=True
    )
    rf = RandomForestClassifier(**{'max_depth': 5}, random_state=rs)
    rf.fit(X_train, y_train)
    y_predict_proba = rf.predict_proba(X_test)[:, 1]

    precision, recall, thresholds2 = precision_recall_curve(y_test, y_predict_proba)
    return thresholds2[np.argmin(abs(precision-recall))]

def get_threshold_list(df, X):
    particles = df.iloc[:, 29:].columns.to_list()
    threshholds = {particle: find_best_threshold(particle, df, X, rs=0) for particle in particles}
    return threshholds