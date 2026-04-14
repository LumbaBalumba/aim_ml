from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def get_train_val_difference(X_train, y_train, X_val, y_val, model_kwargs={}):
    X1 = pd.concat((X_train, y_train), axis=1)
    y1 = np.zeros(X1.shape[0])

    X2 = pd.concat((X_val, y_val), axis=1)
    y2 = np.ones(X2.shape[0])

    X = pd.concat((X1, X2), axis=0)
    y = np.concat((y1, y2), axis=0)

    X_t, X_v, y_t, y_v = train_test_split(X, y)

    train_pool = Pool(X_t, label=y_t, cat_features=model_kwargs.get('cat_features'), embedding_features=model_kwargs.get('embedding_features'))

    val_pool = Pool(X_v, label=y_v, cat_features=model_kwargs.get('cat_features'), embedding_features=model_kwargs.get('embedding_features'))

    model = CatBoostClassifier(**model_kwargs).fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=10,
        early_stopping_rounds=50,)

    y_p = model.predict(X_v)

    return roc_auc_score(y_v, y_p), model, (X_v, y_v)
