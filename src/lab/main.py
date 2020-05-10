from set_logger import create_logger
from stock_optimizer import StockOptimizer
import traceback
import pathlib

path = pathlib.Path(__file__).parent.absolute()
logger = create_logger(f'{path.parent}/results', 'INFO')
overfitting_threshold = 1000  # 0.01, 0.05, 0.1, 0.5 1000
number_of_past_points = 15  # 1, 15
# merece la pena poner number of steps ahead

feed_forward = {
    'model': 'feed_forward_neural_net',
    'training': True,
    'parameters': {
        'optimizer': ['Adam', 'Adagrad'],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'epochs': [3, 10, 30, 50, 70]
    }
}

random_forest = {
    'model': 'random_forest',
    'training': True,
    'parameters': {
        'min_samples_leaf': [1, 2, 4, 8, 10],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': [2, 5, 10, 20, 30],
        'n_estimators': [2, 5, 7, 10, 15, 20, 50, 100, 200, 400],
        'max_depth': [3, 5, 7, 10, 15, 20, 30, 60, 100]
    }
}
"""
random_forest = {
    'model': 'random_forest',
    'training': True,
    'parameters': {
        'min_samples_leaf': [1],
        'max_features': ['auto'],
        'min_samples_split': [2],
        'n_estimators': [200],
        'max_depth': [20]}
}
# trying to reduce ovefitting

random_forest = {
    'model': 'random_forest',
    'training': True,
    'parameters': {
        'min_samples_leaf': [8],
        'max_features': ['auto'],
        'min_samples_split': [2, 10, 17, 20, 25, 40, 50],
        'n_estimators': [2, 4, 8, 20],
        'max_depth': [2, 3, 7, 15, 30]}
}

random_forest = {
    'model': 'random_forest',
    'training': True,
    'parameters': {
        'min_samples_leaf': [4],
        'max_features': ['sqrt'],
        'min_samples_split': [10],
        'n_estimators': [15],
        'max_depth': [5]
    }
}
xgboost = {
    'model': 'xgboost',
    'training': True,
    'parameters': {
        'min_child_weight': [10],
        'gamma': [1.5],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'n_estimators': [15],
        'max_depth': [10]
    }
}

xgboost = {
    'model': 'xgboost',
    'training': True,
    'parameters': {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [2, 5, 7, 10, 15, 20, 50, 100],
        'max_depth': [3, 4, 5, 7, 10, 15, 20]
    }
}
"""
"""
xgboost = {
    'model': 'xgboost',
    'training': False,
    'parameters': {
        'min_child_weight': [1],
        'gamma': [0.5],
        'subsample': [0.6],
        'colsample_bytree': [0.6],
        'n_estimators': [5],
        'max_depth': [3]
    }
}

arima = {
    'model': 'Arima'
}
"""
xgboost = {
    'model': 'xgboost',
    'training': False,
    'parameters': {
        'min_child_weight': [1],
        'gamma': [0.5],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'n_estimators': [50],
        'max_depth': [4]
    }
}

# models = [arima]
# models = [random_forest, xgboost]  # random_forest, feed_forward
models = [random_forest]
# models = [xgboost]


def main():

    stock_optimizer = StockOptimizer(
        number_of_tickers=10,
        models_and_parameters=models,
        overfitting_threshold=overfitting_threshold,
        number_of_past_points=number_of_past_points
    )
    stock_optimizer.run()


if __name__ == '__main__':

    logger.info(f"Program starts here")
    try:
        main()
    except Exception as e:
        track = traceback.format_exc()
        logger.error(f"There was an error while executing the promgram {track}")
