from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
from joblib import dump, load
import pathlib

logger = logging.getLogger('Random Forest')


class RandomForest:

    def __init__(self, ticker):

        self.random_state = 2020
        self.model_path = f'{pathlib.Path(__file__).parent.parent.absolute()}/model'
        self.ticker = ticker

    def __transform_data(self, dataset):

        x = np.empty((0, 77))  # todo: set as parameter
        y = np.empty((0, 1))
        for pos, data in enumerate(dataset):
            x = np.vstack((x, data[0].flatten().reshape(1, -1)))
            y = np.vstack((y, np.array(data[1])))

        return x, y

    def __train(
            self,
            train,
            validation,
            min_samples_leaf,
            max_features,
            min_samples_split,
            n_estimators,
            max_depth):

        train_x, train_y = self.__transform_data(train)
        validation_x, validation_y = self.__transform_data(validation)

        model = RandomForestRegressor(random_state=self.random_state,
                                      min_samples_leaf=min_samples_leaf,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth)

        model.fit(train_x, train_y.flatten())

        predicted_val = model.predict(validation_x)
        predicted_train = model.predict(train_x)

        mse_validation = mean_squared_error(predicted_val, validation_y)
        mse_train = mean_squared_error(predicted_train, train_y)

        dump(
            model,
            f'{self.model_path}/ticker_{self.ticker}_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}'
            f'_min_samples_split_{min_samples_split}_n_estimators_{n_estimators}_max_depth_{max_depth}.joblib')

        return mse_validation, mse_train

    def __load_checkpoint(
            self,
            min_samples_leaf,
            max_features,
            min_samples_split,
            n_estimators,
            max_depth):

        model = load(
            f'{self.model_path}/ticker_{self.ticker}_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}'
            f'_min_samples_split_{min_samples_split}_n_estimators_{n_estimators}_max_depth'
            f'_{max_depth}.joblib')
        return model

    def __test(
            self,
            test,
            min_samples_leaf,
            max_features,
            min_samples_split,
            n_estimators,
            max_depth):

        trained_model = self.__load_checkpoint(
            min_samples_leaf,
            max_features,
            min_samples_split,
            n_estimators,
            max_depth)
        test_x, test_y = self.__transform_data(test)
        predictions_test = trained_model.predict(test_x)

        return predictions_test, test_y

    def __get_trend(self, true_values, predictions):

        ratio = 0
        for pos in range(len(true_values) - 1):

            real_diff = true_values[pos] - true_values[pos + 1]
            prediction_diff = true_values[pos] - predictions[pos + 1]

            if real_diff * prediction_diff > 0:
                ratio += 1

        return ratio / len(true_values) - 1

    def run(self, train, val, test, model_parameters):

        mse = 1000
        best_parameters = {}

        for min_samples_leaf in model_parameters.get(
                'parameters').get('min_samples_leaf'):
            for max_features in model_parameters.get(
                    'parameters').get('max_features'):
                for min_samples_split in model_parameters.get(
                        'parameters').get('min_samples_split'):
                    for n_estimators in model_parameters.get(
                            'parameters').get('n_estimators'):
                        for max_depth in model_parameters.get(
                                'parameters').get('max_depth'):
                            if model_parameters.get('training'):
                                logger.info(
                                    f'Starts Training: min_samples_leaf {min_samples_leaf},'
                                    f' max_features {max_features},'
                                    f' min_samples_split {min_samples_split}, n_estimators {n_estimators}'
                                    f'max_depth {max_depth}')
                                mse_validation, mse_train = self.__train(train,
                                                                         val,
                                                                         min_samples_leaf=min_samples_leaf,
                                                                         max_features=max_features,
                                                                         min_samples_split=min_samples_split,
                                                                         n_estimators=n_estimators,
                                                                         max_depth=max_depth)
                                logger.info(
                                    f'MSE validation {mse_validation} and MSE train {mse_train}')
                            else:
                                mse_validation, mse_train = None, None
                            predictions, true_values = self.__test(
                                test, min_samples_leaf, max_features, min_samples_split,
                                n_estimators, max_depth)
                            logger.info(f'Ends Test')
                            # Todo: check overfitting with: train_loss, val loss
                            current_mse = mean_squared_error(
                                true_values, predictions)
                            if current_mse < mse:
                                best_parameters = {
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features,
                                    'min_samples_split': min_samples_split,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth
                                }
                                mse = current_mse
                                percenatge_of_guess_in_trend = self.__get_trend(
                                    true_values, predictions)
        return best_parameters, mse, percenatge_of_guess_in_trend
