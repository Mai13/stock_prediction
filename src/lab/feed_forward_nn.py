import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import logging
import pathlib
import torch.optim as optim

logger = logging.getLogger('Feed Forward Neural Net')


class Net(nn.Module):

    def __init__(self, dimension_of_first_layer):
        super().__init__()
        self.fc1 = nn.Linear(
            dimension_of_first_layer,
            100)  # number_of_past_points
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class FeedForwardNN:

    def __init__(
            self,
            dimension_of_first_layer,
            number_of_epoch,
            learning_rate_adam,
            ticker):
        self.ticker = ticker
        self.number_of_epoch = number_of_epoch
        self.net = Net(dimension_of_first_layer)
        self.lr_adam = learning_rate_adam
        self.model_path = f'{pathlib.Path(__file__).parent.parent.absolute()}/model'

    def __train(self, optimizer, criterion, train, validation):

        train_loss = []
        val_loss = []
        for epoch in range(self.number_of_epoch):
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
            f'{self.model_path}/{self.ticker}_learning_rate_{self.lr_adam}_epochs_{self.number_of_epoch}.pth')
        return train_loss, val_loss

    def __load_checkpoint(self):

        print(f'{self.model_path}/{self.ticker}_learning_rate_{self.lr_adam}_epochs_{self.number_of_epoch}.pth')
        checkpoint = torch.load(
            f'{self.model_path}/{self.ticker}_learning_rate_{self.lr_adam}_epochs_{self.number_of_epoch}.pth')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def __test(self, test):

        trained_model = self.__load_checkpoint()
        predictions = []
        true_values = []

        for test_data in test:
            X, y = test_data
            output = trained_model(torch.Tensor(X.flatten()))
            predictions.append(output)
            true_values.append(y)

        return predictions, true_values

    def run(self, train, val, test):

        logger.info(
            f'Starts Training ADAM with learning_rate {self.lr_adam}, epochs {self.number_of_epoch}')
        train_loss, val_loss = self.__train(optimizer=optim.Adam(self.net.parameters(), lr=self.lr_adam),  # Aqui tendrias que cambiar el optimizer
                                            criterion=torch.nn.MSELoss(),
                                            train=train,
                                            validation=val)
        logger.info(f'Ends Training ADAM')
        logger.info(f'Starts Test')
        predictions, true_values = self.__test(test)
        logger.info(f'Ends Test')

        return predictions, true_values, train_loss, val_loss

# create test
