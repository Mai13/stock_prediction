import torch.nn as nn
import torch.nn.functional as F


"""
data_array = data.values
data_array = data_array[np.argsort(-1*data_array[:,2])]
print(data_array)
number_of_past_points = 4
transformed_data = []
for position in range(1, data_array.shape[0]-number_of_past_points):
    transformed_data.append([data_array[position: position+number_of_past_points, 1].astype(int), int(data_array[position-1, 1])])
X, y = transformed_data[0]
print(X, y)
transformed_data.__len__()
print(transformed_data)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
epochs = 10 # Todo : This parameter must be higher (1000)
loss_func = torch.nn.MSELoss()

loss_adam_train = []
loss_adam_val = []

mse_adam_train = []
mse_adam_val = []

epochs_adam = []

for epoch in range(epochs):
  val_mse_for_mean = []
  for data in train:
    X, y = data
    output = net(torch.Tensor(X))
    loss = loss_func(output, torch.Tensor(np.array([y])))
    loss.backward()
    optimizer.step()
    val_mse_for_mean.append(float(((output - y)**2).mean().detach().numpy()))
  loss_adam_train.append(loss)
  mse_adam_train.append(mean(val_mse_for_mean))
  net.zero_grad()
  with torch.no_grad():
    val_mse_for_mean = []
    for val_data in validation:
      X, y = val_data
      output = net(torch.Tensor(X))
      loss_val = loss_func(output, torch.Tensor(np.array([y])))
      val_mse_for_mean.append(float(((output - y)**2).mean().detach().numpy()))
    mse_adam_val.append(mean(val_mse_for_mean))
    loss_adam_val.append(loss_val)
  print(f'training epoch {epoch}: loss {loss}')
  
  epochs_adam.append(epoch)
"""

class Net(nn.Module):

    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(10, 10) # number_of_past_points
      self.fc2 = nn.Linear(10, 10)
      self.fc3 = nn.Linear(10, 10)
      self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.relu(x)


# create test