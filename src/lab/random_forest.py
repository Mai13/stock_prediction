from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
from joblib import dump, load
import pathlib
import os

logger = logging.getLogger('Random Forest')


class RandomForest:

    def __init__(self, ticker, overfitting_threshold):

        self.random_state = 2020
        self.model_path = f'{pathlib.Path(__file__).parent.parent.absolute()}/model'
        self.ticker = ticker
        self.overfitting_threshold = overfitting_threshold

    def __transform_data(self, dataset):

        first_element, second_element = dataset[0]
        # print(first_element.shape[0]*first_element.shape[1])
        # todo: set as parameter
        x = np.empty((0, first_element.shape[0] * first_element.shape[1]))
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

        if abs(mse_validation - mse_train) < self.overfitting_threshold:
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

        model_path = f'{self.model_path}/ticker_{self.ticker}_min_samples_leaf_{min_samples_leaf}_max_features_' \
                     f'{max_features}_min_samples_split_{min_samples_split}_n_estimators_{n_estimators}_max_depth' \
                     f'_{max_depth}.joblib'

        if os.path.exists(model_path):
            model = load(model_path)
        else:
            model = None
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

        if trained_model:
            test_x, test_y = self.__transform_data(test)
            predictions_test = trained_model.predict(test_x)
            there_is_prediction = True
        else:
            logger.info(
                'This model does not exist due to the overfitting threshold')
            predictions_test, test_y = None, None
            there_is_prediction = False
        return predictions_test, test_y, there_is_prediction

    def __get_trend(self, true_values, predictions):

        ratio = 0

        for pos in range(len(true_values) - 1):

            real_diff = true_values[pos] - true_values[pos + 1]
            prediction_diff = true_values[pos] - predictions[pos + 1]

            if real_diff * prediction_diff > 0:
                ratio += 1

        return ratio / (len(true_values) - 1)

    def run(self, train, val, test, model_parameters):

        mse = 1000
        mse_val = 1000
        best_parameters = {}
        percenatge_of_guess_in_trend = 0
        best_prediction = 0
        true_values_list = []
        there_is_a_best_prediction = False

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
                                    f'MSE validation {mse_validation} and MSE train {mse_train},'
                                    f' diff {abs(mse_validation-mse_train)}')
                            else:
                                mse_validation, mse_train = None, None
                            predictions, true_values, there_is_prediction = self.__test(
                                test, min_samples_leaf, max_features, min_samples_split, n_estimators, max_depth)
                            """
                            if there_is_prediction:
                                current_mse = mean_squared_error(
                                    true_values, predictions)
                                logger.info(f'test MSE: {mse}, MSE validation {mse_validation}, MSE train {mse_train}')
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
                                    best_prediction = predictions
                                    there_is_a_best_prediction = True
                                    true_values_list = true_values
                                    logger.info(f'BEST test mse: {mse}')

                            """
                            if there_is_prediction:
                                current_mse = mean_squared_error(
                                    true_values, predictions)
                                logger.info(
                                    f'test MSE: {mse}, MSE validation {mse_validation}, MSE train {mse_train}')
                                if mse_validation < mse_val:
                                    best_parameters = {
                                        'min_samples_leaf': min_samples_leaf,
                                        'max_features': max_features,
                                        'min_samples_split': min_samples_split,
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth
                                    }
                                    mse_val = mse_validation
                                    percenatge_of_guess_in_trend = self.__get_trend(
                                        true_values, predictions)
                                    best_prediction = predictions
                                    there_is_a_best_prediction = True
                                    true_values_list = true_values
                                    logger.info(
                                        f'BEST test mse: {mse}, val mse {mse_validation}')

        return best_parameters, mse, percenatge_of_guess_in_trend, best_prediction, true_values_list, there_is_a_best_prediction
