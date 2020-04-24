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
        a = train[:, 0][0].reshape(-1, 1)
        print(train[:, 0].shape, train[:, 0].reshape(-1, 1).shape, train[:, 0].reshape(1, -1).shape)
        model.fit(train[:, 0].reshape(-1, 1)[0], train[:, 1])

        predicted_val = model.predict(validation[:, 0])
        predicted_train = model.predict(train[:, 0])

        mse_validation = mean_squared_error(predicted_val, validation[:, 1])
        mse_train = mean_squared_error(predicted_train, train[:, 1])

        dump(model, f'{self.model_path}/ticker_{self.ticker}_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}_criterion_{criterion}'
                    f'_min_samples_split_{min_samples_split}_n_estimators_{n_estimators}_max_depth_{max_depth}.joblib')

        return mse_validation, mse_train

    def __load_checkpoint(self, min_samples_leaf, max_features, criterion, min_samples_split, n_estimators, max_depth):

        model = load(f'{self.model_path}/ticker_{self.ticker}_min_samples_leaf_{min_samples_leaf}_max_features_{max_features}_criterion_'
                     f'{criterion}_min_samples_split_{min_samples_split}_n_estimators_{n_estimators}_max_depth'
                     f'_{max_depth}.joblib')
        return model

    def __test(self, test, min_samples_leaf, max_features, criterion, min_samples_split, n_estimators, max_depth):

        trained_model = self.__load_checkpoint(min_samples_leaf, max_features, criterion,
                                               min_samples_split, n_estimators, max_depth)
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
                                    mse_validation, mse_train = self.__train(train,
                                                                             val,
                                                                             min_samples_leaf=min_samples_leaf,
                                                                             max_features=max_features,
                                                                             criterion=criterion,
                                                                             min_samples_split=min_samples_split,
                                                                             n_estimators=n_estimators,
                                                                             max_depth=max_depth)
                                    logger.info(f'MSE validation {mse_validation} and MSE train {mse_train}')
                                else:
                                    mse_validation, mse_train = None, None
                                predictions, true_values = self.__test(
                                    test, min_samples_leaf, max_features, criterion, min_samples_split,
                                    n_estimators, max_depth)
                                logger.info(f'Ends Test')
                                # Todo: check overfitting with: train_loss, val_loss
                                current_mse = mean_squared_error(true_values, predictions)
                                if current_mse < mse:
                                    best_parameters = {
                                        'min_samples_leaf': min_samples_leaf,
                                        'max_features': max_features,
                                        'criterion': criterion,
                                        'min_samples_split': min_samples_split,
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth
                                    }
                                    mse = current_mse
                                percenatge_of_guess_in_trend = self.__get_trend(
                                    true_values, predictions)
        return best_parameters, mse, percenatge_of_guess_in_trend
