from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger('Random Forest')


class RandomForest:

    def __train(self):
        pass

    def run(self, train, val, test, model_parameters):

        mse = 0
        best_parameters = {}

        for optimizer_name in model_parameters.get(
                'parameters').get('optimizer'):
            for epoch in model_parameters.get('parameters').get('epochs'):
                for learning_rate in model_parameters.get(
                        'parameters').get('learning_rate'):
                    if model_parameters.get('training'):
                        logger.info(
                            f'Starts Training: learning_rate {learning_rate}, epochs {epoch}, optimzer {optimizer_name}')
                        if optimizer_name == 'Adam':
                            train_loss, val_loss = self.__train()
                        elif optimizer_name == 'Adagrad':
                            train_loss, val_loss = self.__train()
                        logger.info(f'Ends Training')
                    else:
                        train_loss, val_loss = None, None
                    logger.info(
                        f'Starts Test:  learning_rate {learning_rate}, epochs {epoch}, optimzer {optimizer_name}')
                    predictions, true_values = self.__test(
                        test, optimizer_name, learning_rate, epoch)
                    logger.info(f'Ends Test')
                    self.graphing_module.plot_overfitting_graph(
                        train_loss, val_loss, epoch, optimizer_name, learning_rate, self.ticker)
                    self.graphing_module.plot_test_graph(
                        predictions, true_values, epoch, optimizer_name, learning_rate, self.ticker)
                    # Todo: check overfitting with: train_loss, val_loss
                    current_mse = mean_squared_error(true_values, predictions)
                    if current_mse < mse:
                        best_parameters = {
                            'learning_rate': learning_rate,
                            'epochs': epoch,
                            'optimizer': optimizer_name,
                        }
                    percenatge_of_guess_in_trend = self.__get_trend(
                        true_values, predictions)
        return best_parameters, mse, percenatge_of_guess_in_trend
