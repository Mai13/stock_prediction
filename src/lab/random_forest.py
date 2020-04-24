from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import logging

logger = logging.getLogger('Random Forest')


class RandomForest:

    def __init__(self):

        self.random_state = 2020

    def __train(self, train, validation, min_samples_leaf, max_features, criterion, min_samples_split, n_estimators, max_depth):
        train = np.array(train)
        validation = np.array(validation)
        model = RandomForestRegressor(random_state=self.random_state,
                                      min_samples_leaf=min_samples_leaf,
                                      max_features=max_features,
                                      criterion=criterion,
                                      min_samples_split=min_samples_split,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth)
        model.fit(train[:, 0], train[:, 1])
        predicted_val = model.predict(validation[:, 0])
        predicted_train = model.predict(train[:, 0])

        mse_validation = mean_squared_error(predicted_val, validation[:, 1])
        mse_train = mean_squared_error(predicted_train, train[:, 1])

        return mse_validation, mse_train

    def __load_checkpoint(self, optimizer, learning_rate, epoch):

        model = 0  # TODO: load the modle with sklearn
        return model

    def __test(self, test, optimizer, learning_rate, epoch):

        trained_model = self.__load_checkpoint(optimizer, learning_rate, epoch)
        predictions = []
        true_values = []

        for test_data in test:
            X, y = test_data
            output = trained_model(X)
            predictions.append(output)
            true_values.append(y)

        return predictions, true_values

    def __get_trend(self, true_values, predictions):

        ratio = 0
        for pos in range(len(true_values) - 1):

            real_diff = true_values[pos] - true_values[pos + 1]
            prediction_diff = true_values[pos] - predictions[pos + 1]

            if real_diff * prediction_diff > 0:
                ratio += 1

        return ratio / len(true_values) - 1

    def run(self, train, val, test, model_parameters):

        mse = 0
        best_parameters = {}

        for min_samples_leaf in model_parameters.get('parameters').get('min_samples_leaf'):
            for max_features in model_parameters.get('parameters').get('max_features'):
                for criterion in model_parameters.get('parameters').get('criterion'):
                    for min_samples_split in model_parameters.get('parameters').get('min_samples_split'):
                        for n_estimators in model_parameters.get('parameters').get('n_estimators'):
                            for max_depth in model_parameters.get('parameters').get('max_depth'):
                                if model_parameters.get('training'):
                                    logger.info(f'Starts Training: min_samples_leaf {min_samples_leaf},'
                                                f' max_features {max_features}, criterion {criterion},'
                                                f' min_samples_split {min_samples_split}, n_estimators {n_estimators}'
                                                f'max_depth {max_depth}')
                                    mse_validation, mse_train = self.__train(min_samples_leaf=min_samples_leaf,
                                                                        max_features=max_features,
                                                                        criterion=criterion,
                                                                        min_samples_split=min_samples_split,
                                                                        n_estimators=n_estimators,
                                                                        max_depth=max_depth)
                                    logger.info(f'MSE validation {mse_validation} and MSE train {mse_train}')
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
