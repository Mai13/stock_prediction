import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import logging
import pathlib
import torch.optim as optim
from graph_maker import CreateGraphs
from sklearn.metrics import mean_squared_error

logger = logging.getLogger('Feed Forward Neural Net')


class Net(nn.Module):

    def __init__(self, dimension_of_first_layer):
        super().__init__()
        self.fc1 = nn.Linear(
            dimension_of_first_layer, 10)  # number_of_past_points
        self.fc2 = nn.Linear(10, 10)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class FeedForwardNN:

    def __init__(self, dimension_of_first_layer, ticker):

        self.ticker = ticker
        self.net = Net(dimension_of_first_layer)
        self.model_path = f'{pathlib.Path(__file__).parent.parent.absolute()}/model'
        self.graphing_module = CreateGraphs()

    def __train(
            self,
            optimizer,
            criterion,
            train,
            validation,
            epochs,
            learning_rate,
            optimizer_name):

        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            loss_accum = 0.0
            for data in train:
                X, y = data
                optimizer.zero_grad()
                # print(torch.Tensor(X.flatten()).shape)
                output = self.net(torch.Tensor(X.flatten()))
                loss = criterion(output, torch.Tensor(np.array([y])))
                loss_accum += loss.data.item()
                loss.backward()
                optimizer.step()

            loss_accum_val = 0.0
            for val_data in validation:
                val_x, val_y = val_data
                output = self.net(torch.Tensor(val_x.flatten()))
                loss_accum_val += criterion(output,
                                            torch.Tensor(np.array([val_y]))).data.item()

            logger.info(
                f'Epoch:, {epoch}, "train loss:", {loss_accum/len(train)}, "val loss:", {loss_accum_val/len(validation)}')
            train_loss.append(loss_accum / len(train))
            val_loss.append(loss_accum_val / len(validation))

        checkpoint = {'model': self.net,
                      'state_dict': self.net.state_dict(),
                      'optimizer': optimizer.state_dict()}

        torch.save(
            checkpoint,
            f'{self.model_path}/{self.ticker}_optimizer_{optimizer_name}_learning_rate_{learning_rate}_epochs_{epochs}.pth')
        return train_loss, val_loss

    def __load_checkpoint(self, optimizer, learning_rate, epoch):

        checkpoint = torch.load(
            f'{self.model_path}/{self.ticker}_optimizer_{optimizer}_learning_rate_{learning_rate}_epochs_{epoch}.pth')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def __test(self, test, optimizer, learning_rate, epoch):

        trained_model = self.__load_checkpoint(optimizer, learning_rate, epoch)
        predictions = []
        true_values = []

        for test_data in test:
            X, y = test_data
            output = trained_model(torch.Tensor(X.flatten()))
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

        mse = 1000
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
                            train_loss, val_loss = self.__train(
                                optimizer=optim.Adam(self.net.parameters(), lr=learning_rate),
                                criterion=torch.nn.MSELoss(),
                                train=train,
                                validation=val,
                                epochs=epoch,
                                learning_rate=learning_rate,
                                optimizer_name=optimizer_name
                            )
                        elif optimizer_name == 'Adagrad':
                            train_loss, val_loss = self.__train(
                                optimizer=optim.Adagrad(self.net.parameters(), lr=learning_rate),
                                criterion=torch.nn.MSELoss(),
                                train=train,
                                validation=val,
                                epochs=epoch,
                                learning_rate=learning_rate,
                                optimizer_name=optimizer_name
                            )
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
