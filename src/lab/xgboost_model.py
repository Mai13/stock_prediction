import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
from joblib import dump, load
import pathlib
import os
import pickle

logger = logging.getLogger('XGBoost')


class XGBoost:

    def __init__(self, ticker, overfitting_threshold):

        self.random_state = 2020
        self.model_path = f'{pathlib.Path(__file__).parent.parent.absolute()}/model'
        self.ticker = ticker
        self.overfitting_threshold = overfitting_threshold

    def __transform_data(self, dataset):

        first_element, second_element = dataset[0]
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
            min_child_weight,
            gamma,
            subsample,
            colsample_bytree,
            n_estimators,
            max_depth):

        is_overfitted = True
        train_x, train_y = self.__transform_data(train)
        validation_x, validation_y = self.__transform_data(validation)

        model = xgb.XGBRegressor(min_child_weight=min_child_weight,
                                 gamma=gamma,
                                 subsample=subsample,
                                 colsample_bytree=colsample_bytree,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth)

        model.fit(train_x, train_y.flatten())

        predicted_val = model.predict(validation_x)
        predicted_train = model.predict(train_x)

        mse_validation = mean_squared_error(predicted_val, validation_y)
        mse_train = mean_squared_error(predicted_train, train_y)

        if abs(mse_validation - mse_train) < self.overfitting_threshold:
            is_overfitted = False

        return mse_validation, mse_train, is_overfitted, model

    def __load_checkpoint(
            self,
            min_child_weight,
            gamma,
            subsample,
            colsample_bytree,
            n_estimators,
            max_depth):

        model_path = f'{self.model_path}/ticker_{self.ticker}_min_child_weight_{min_child_weight}_gamma_{gamma}' \
                     f'_subsample_{subsample}_colsample_bytree_{colsample_bytree}_n_estimator_' \
                     f'{n_estimators}_max_depth_{max_depth}.joblib'

        if os.path.exists(model_path):
            model = load(model_path)
        else:
            model = None
        return model

    def __test(
            self,
            test,
            min_child_weight,
            gamma,
            subsample,
            colsample_bytree,
            n_estimators,
            max_depth):

        trained_model = self.__load_checkpoint(
            min_child_weight,
            gamma,
            subsample,
            colsample_bytree,
            n_estimators,
            max_depth)

        if trained_model:
            test_x, test_y = self.__transform_data(test)
            predictions_test = trained_model.predict(test_x)
            there_is_prediction = True
        else:
            logger.info('This model does not exist due to the overfitting threshold')
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

        mse_val = 1000
        there_is_a_best_prediction = False

        for min_child_weight in model_parameters.get('parameters').get('min_child_weight'):
            for gamma in model_parameters.get('parameters').get('gamma'):
                for subsample in model_parameters.get('parameters').get('subsample'):
                    for colsample_bytree in model_parameters.get('parameters').get('colsample_bytree'):
                        for n_estimators in model_parameters.get('parameters').get('n_estimators'):
                            for max_depth in model_parameters.get('parameters').get('max_depth'):
                                if model_parameters.get('training'):
                                    logger.info(
                                        f'Starts Training: min_child_weight {min_child_weight},'
                                        f' gamma {gamma}, subsample {subsample}'
                                        f' colsample_bytree {colsample_bytree}, n_estimators {n_estimators}'
                                        f'max_depth {max_depth}')
                                    mse_validation, mse_train, is_overfitted, model = self.__train(train,
                                                                                                     val,
                                                                                                     min_child_weight=min_child_weight,
                                                                                                     gamma=gamma,
                                                                                                     subsample=subsample,
                                                                                                     colsample_bytree=colsample_bytree,
                                                                                                     n_estimators=n_estimators,
                                                                                                     max_depth=max_depth)
                                    logger.info(
                                        f'MSE validation {mse_validation} and MSE train {mse_train}'
                                        f' diff {abs(mse_validation-mse_train)}')
                                    if not is_overfitted:
                                        if mse_validation < mse_val:
                                            best_parameters = {
                                                'min_child_weight': min_child_weight,
                                                'gamma': gamma,
                                                'subsample': subsample,
                                                'colsample_bytree': colsample_bytree,
                                                'n_estimators': n_estimators,
                                                'max_depth': max_depth
                                            }
                                            dump(model, f'{self.model_path}/ticker_{self.ticker}_min_child_weight_'
                                                        f'{min_child_weight}_gamma_{gamma}_subsample_{subsample}_'
                                                        f'colsample_bytree_{colsample_bytree}_n_estimator_'
                                                        f'{n_estimators}_max_depth_{max_depth}.joblib')
                                            pickle.dump(best_parameters, open(f'{self.model_path}/ticker_{self.ticker}_{model_parameters.get("model")}_best_model.p', 'wb'))
                                            there_is_a_best_prediction = True
        if there_is_a_best_prediction:
            best_parameters = pickle.load(open(f'{self.model_path}/ticker_{self.ticker}_{model_parameters.get("model")}_best_model.p', 'rb'))
            predictions, true_values, there_is_prediction = self.__test(test, best_parameters.get('min_child_weight'),
                                                                        best_parameters.get('gamma'),
                                                                        best_parameters.get('subsample'),
                                                                        best_parameters.get('colsample_bytree'),
                                                                        best_parameters.get('n_estimators'),
                                                                        best_parameters.get('max_depth'))
            current_mse = mean_squared_error(true_values, predictions)
            logger.info(f'Best {model_parameters.get("model")} for {self.ticker}'
                        f' test mse: {current_mse},'
                        f' validation mse: {model_parameters.get("mse_validation")},'
                        f' train mse: {model_parameters.get("mse_train")}')
            percenatge_of_guess_in_trend = self.__get_trend(true_values, predictions)
        else:
            logger.ERROR(f'Best models Model file is missing')
        return best_parameters, current_mse, percenatge_of_guess_in_trend, predictions, true_values, there_is_prediction
