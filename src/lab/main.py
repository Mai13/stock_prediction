from set_logger import create_logger
from stock_optimizer import StockOptimizer
import traceback
import pathlib

path = pathlib.Path(__file__).parent.absolute()
logger = create_logger(f'{path.parent}/results', 'INFO')

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
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [2, 5, 7, 10, 15, 20, 50, 100, 200, 400],
        'max_depth': [3, 5, 7, 10, 15, 20, 30, 60, 100]
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
arima = {
    'model': 'Arima'
}

# models = [arima]
models = [xgboost]


def main():

    stock_optimizer = StockOptimizer(
        number_of_tickers=10,
        models_and_parameters=models)
    stock_optimizer.run()


if __name__ == '__main__':

    logger.info(f"Program starts here")
    try:
        main()
    except Exception as e:
        track = traceback.format_exc()
        logger.error(
            f"There was an error while executing the promgram {track}")
