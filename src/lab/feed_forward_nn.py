import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import logging
import pathlib

logger = logging.getLogger('Feed Forward Neural Net')


class Net(nn.Module):

    def __init__(self, dimension_of_first_layer):
        super().__init__()
        self.fc1 = nn.Linear(dimension_of_first_layer, 10)  # number_of_past_points
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class FeedForwardNN:

    def __init__(self, number_of_previous_points, number_of_epoch, dataset, learning_rate_adam):
        self.number_of_previous_points = number_of_previous_points
        self.number_of_epoch = number_of_epoch
        self.dataset = dataset
        self.net = Net(self.number_of_previous_points)
        self.lr_adam = learning_rate_adam
        self.model_path = f'{pathlib.Path(__file__).parent.parent.absolute()}/model/'

    def __train(self, optimizer, criterion, train, validation):

        train_loss = []
        val_loss = []
        for epoch in range(self.number_of_epoch):
            loss_accum = 0.0
            for data in train:
                X, y = data
                optimizer.zero_grad()
                output = self.net(torch.Tensor(X))
                loss = criterion(output, torch.Tensor(np.array([y])))
                loss_accum += loss.data.item()
                loss.backward()
                optimizer.step()

            loss_accum_val = 0.0
            for val_data in validation:
                val_x, val_y = val_data
                output = self.net(torch.Tensor(val_x))
                loss_accum_val += criterion(output, torch.Tensor(np.array([val_y]))).data.item()

            logger.info(f'Epoch:, {epoch}, "train loss:", {loss_accum/len(train)}, "val loss:", {loss_accum_val/len(validation)}')
            train_loss.append(loss_accum / len(train))
            val_loss.append(loss_accum_val / len(validation))

        checkpoint = {'model': self.net,
                      'state_dict': self.net.state_dict(),
                      'optimizer': optimizer.state_dict()}

        torch.save(checkpoint, f'{self.model_path}/{self.dataset}_learning_rate_{self.lr_adam}_epochs_{self.number_of_epoch}.pth')
        return train_loss, val_loss

    def __load_checkpoint(self):

        checkpoint = torch.load(f'{self.model_path}/{self.dataset}_learning_rate_{self.lr_adam}_epochs_{self.number_of_epoch}.pth')
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
            output = trained_model(torch.Tensor(X))
            predictions.append(output)
            true_values.append(y)

        return predictions, true_values

# create test
