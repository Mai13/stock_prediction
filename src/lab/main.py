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
models = [feed_forward]


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
