from set_logger import create_logger
from stock_optimizer import StockOptimizer
import traceback
import pathlib

path = pathlib.Path(__file__).parent.absolute()
logger = create_logger(f'{path.parent}/results', 'INFO')

"""
            training_parameters = {'model': 'FeedForwardNerualNet',
                                   'trainig': True | False,
                                   'parameters': {'Optimizer': 'Adam'/'Adagrad',
                                                  'learning_rate': [0.1, 0.01],
                                                  'epochs': [10, 40, 80, 100]
                                                }}
            """


def main():

    stock_optimizer = StockOptimizer(number_of_tickers=10, train=True)
    stock_optimizer.run()


if __name__ == '__main__':

    logger.info(f"Program starts here")
    try:
        main()
    except Exception as e:
        track = traceback.format_exc()
        logger.error(
            f"There was an error while executing the promgram {track}")
