import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

import os
os.chdir('../..')


from joblib import dump

from src.features.build_features import build_features
from src.data.preprocessor import create_preprocessor
from src.utils.paths import MODELS_DIR

def train(path):
    df = pd.read_csv(path)

    df_builded = build_features(df)

    X = df_builded.drop('log_charges', axis=1)
    y = df_builded['log_charges']

    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor(numeric_cols, categorical_cols)

    models = [LinearRegression, Lasso, Ridge, ElasticNet, RandomForestRegressor]

    names = []
    train = []
    test = []
    train_scores = []
    test_scores = []
    diffs = []
    comments = []

    for model in models:
        baseline_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model())
        ])

        baseline_pipeline.fit(X_train, y_train)

        train_score = baseline_pipeline.score(X_train, y_train)
        test_score = baseline_pipeline.score(X_test, y_test)
        diff = abs(train_score - test_score)

        names.append(f'{model.__name__}')
        train.append(train_score)
        test.append(test_score)
        train_scores.append(True if train_score >= 0.85 else False)
        test_scores.append(True if test_score >= 0.8 else False)
        diffs.append(True if diff <= 0.05 else False)
        comments.append(True if train_score >= 0.85 and test_score >= 0.8 and diff <= 0.05 else False)

    report_df = pd.DataFrame({
        'Model name': names,
        'Train score': train,
        'Test score': test,
        'Good train?': train_scores,
        'Good test?': test_scores,
        'Good diff?': diffs,
        'Comment': comments
    })

    print(report_df)

    best_model = ''
    pos = 0
    for name, comment in report_df[['Model name', 'Comment']].values:
        if comment:
            best_model = name
            break
        pos += 1
        if pos == 5:
            pos = 0
            max_test_score = max(report_df['Test score'])
            for name, test_score in report_df[['Model name', 'Test score']].values:
                if test_score == max_test_score:
                    best_model = name
                    break
    
    grid_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', models[pos]())
    ])

    params = {
        'LinearRegression': {
            'regressor__fit_intercept': [True, False],
            'regressor__copy_X': [True, False],
            'regressor__n_jobs': [-1]
        },
        'Lasso': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
            'regressor__max_iter': [1000, 2500, 5000],
            'regressor__tol': [0.001, 0.01, 0.1, 1.0]
        },
        'Ridge': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr'],
            'regressor__max_iter': [1000, 2500, 5000]
        },
        'ElasticNet': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
            'regressor__l1_ratio': [0.25, 0.5, 0.75, 1.0],
            'regressor__max_iter': [2000, 5000, 10000]
        },
        'RandomForestRegressor': {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__max_depth': [3, 5, 7, 10, None],
            'regressor__min_samples_split': [2, 5, 10, 20]
        }
    }

    grid_search = GridSearchCV(
        estimator=grid_pipe,
        param_grid=params[best_model],
        scoring='r2',
        verbose=0,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    y_pred_best = grid_search.best_estimator_.predict(X_test)

    r2_best = r2_score(y_test, y_pred_best)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = root_mean_squared_error(y_test, y_pred_best)

    print('Качество модели на обучающей выборке после подбора гиперпараметров:', grid_search.score(X_train, y_train))
    print('Качество модели на тестовой выборке после подбора гиперпараметров:', grid_search.score(X_test, y_test))
    print('Разница:', abs(grid_search.score(X_train, y_train) - grid_search.score(X_test, y_test)))
    print('R2 score:', r2_best)
    print('MAE:', mae_best)
    print('MSE:', mse_best)
    print('RMSE:', rmse_best)

    dump(grid_search.best_estimator_, MODELS_DIR / 'model.joblib')
    return df_builded