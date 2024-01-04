import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('first_reg_stats.csv', usecols=['xwoba', 'exit_velocity_avg', 'launch_angle_avg', 'hard_hit_percent', 'sprint_speed'])

# Define the predictors and target variable
predictors = ['exit_velocity_avg', 'launch_angle_avg', 'hard_hit_percent', 'sprint_speed']
target = 'xwoba'

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Define the neural network
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Linear(len(predictors), 15)
        self.hidden2 = nn.Linear(15, 10)
        self.hidden3 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x

model = RegressionModel()

# Define the loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(train_data.values):
        inputs = torch.tensor(data[1:], dtype=torch.float32)
        labels = torch.tensor(data[0], dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            print(f'Epoch {epoch+1}: loss is NaN')
            break
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if torch.isnan(loss):
        break

    print(f'Epoch {epoch+1}: loss={running_loss/len(train_data):.4f}')

    if running_loss/len(train_data) < 0.0001:
        break
torch.save(model.state_dict(), 'xwoba_reg_mod.pt')
# Predict xWOBA for the test set
inputs = torch.tensor(test_data[predictors].values, dtype=torch.float32)
predictions = model(inputs).detach().numpy()

# Evaluate the model
score = r2_score(test_data[target], predictions)
print(f'The R-squared score of the model is {score:.2f}.')


