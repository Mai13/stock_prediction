from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
from joblib import dump, load
import pathlib
import os
import pickle

logger = logging.getLogger('Regressors')


class Regressors:

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

    def __train_for_regressors(self):

        estimators = [('OLS', LinearRegression()),
                      ('Theil-Sen', TheilSenRegressor(random_state=42)),
                      ('RANSAC', RANSACRegressor(random_state=42)),
                      ('HuberRegressor', HuberRegressor())]
        colors = {
            'OLS': 'turquoise',
            'Theil-Sen': 'gold',
            'RANSAC': 'lightgreen',
            'HuberRegressor': 'black'}
        linestyle = {
            'OLS': '-',
            'Theil-Sen': '-.',
            'RANSAC': '--',
            'HuberRegressor': '--'}
        lw = 3

        x_plot = np.linspace(X.min(), X.max())
        for title, this_X, this_y in [
            ('Modeling Errors Only', X, y),
            ('Corrupt X, Small Deviants', X_errors, y),
            ('Corrupt y, Small Deviants', X, y_errors),
            ('Corrupt X, Large Deviants', X_errors_large, y),
                ('Corrupt y, Large Deviants', X, y_errors_large)]:
            plt.figure(figsize=(5, 4))
            plt.plot(this_X[:, 0], this_y, 'b+')

            for name, estimator in estimators:
                model = make_pipeline(PolynomialFeatures(3), estimator)
                model.fit(this_X, this_y)
                mse = mean_squared_error(model.predict(X_test), y_test)
                y_plot = model.predict(x_plot[:, np.newaxis])
                plt.plot(
                    x_plot,
                    y_plot,
                    color=colors[name],
                    linestyle=linestyle[name],
                    linewidth=lw,
                    label='%s: error = %.3f' %
                    (name,
                     mse))

            legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
            legend = plt.legend(
                loc='upper right',
                frameon=False,
                title=legend_title,
                prop=dict(
                    size='x-small'))
            plt.xlim(-4, 10.2)
            plt.ylim(-2, 10.2)
            plt.title(title)
        plt.show()

    def __train(
            self,
            train,
            validation,
            min_samples_leaf,
            max_features,
            min_samples_split,
            n_estimators,
            max_depth):

        is_overfitted = True
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
            is_overfitted = False

        return mse_validation, mse_train, is_overfitted, model

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

        mse_val = 1000
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
                                mse_validation, mse_train, is_overfitted, model = self.__train(train,
                                                                                               val,
                                                                                               min_samples_leaf=min_samples_leaf,
                                                                                               max_features=max_features,
                                                                                               min_samples_split=min_samples_split,
                                                                                               n_estimators=n_estimators,
                                                                                               max_depth=max_depth)
                                logger.info(
                                    f'MSE validation {mse_validation} and MSE train {mse_train},'
                                    f' diff {abs(mse_validation-mse_train)}')
                                if not is_overfitted:
                                    if mse_validation < mse_val:
                                        best_parameters = {
                                            'min_samples_leaf': min_samples_leaf,
                                            'max_features': max_features,
                                            'min_samples_split': min_samples_split,
                                            'n_estimators': n_estimators,
                                            'max_depth': max_depth,
                                            'mse_validation': mse_validation,
                                            'mse_train': mse_train}
                                        dump(
                                            model,
                                            f'{self.model_path}/ticker_{self.ticker}_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}'
                                            f'_min_samples_split_{min_samples_split}_n_estimators_{n_estimators}_max_depth_{max_depth}.joblib')
                                        pickle.dump(
                                            best_parameters,
                                            open(
                                                f'{self.model_path}/ticker_{self.ticker}_{model_parameters.get("model")}_best_model.p',
                                                'wb'))
                                        there_is_a_best_prediction = True
        if there_is_a_best_prediction:
            best_parameters = pickle.load(
                open(
                    f'{self.model_path}/ticker_{self.ticker}_{model_parameters.get("model")}_best_model.p',
                    'rb'))
            predictions, true_values, there_is_prediction = self.__test(test, best_parameters.get('min_samples_leaf'),
                                                                        best_parameters.get('max_features'),
                                                                        best_parameters.get('min_samples_split'),
                                                                        best_parameters.get('n_estimators'),
                                                                        best_parameters.get('max_depth'))
            current_mse = mean_squared_error(true_values, predictions)
            logger.info(
                f'Best {model_parameters.get("model")} for {self.ticker}'
                f' test mse: {current_mse},'
                f' validation mse: {model_parameters.get("mse_validation")},'
                f' train mse: {model_parameters.get("mse_train")}')
            percenatge_of_guess_in_trend = self.__get_trend(
                true_values, predictions)

        else:
            logger.error(f'Best models Model file is missing')
        return best_parameters, current_mse, percenatge_of_guess_in_trend, predictions, true_values, there_is_prediction
