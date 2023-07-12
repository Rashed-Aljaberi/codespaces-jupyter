import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split,DataLoader, Dataset
from codecarbon import track_emissions
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import matplotlib.pyplot as plt

print('start of cnn') 
# Load and normalize the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(mnist_data))
test_size = len(mnist_data) - train_size
train_data, test_data = random_split(mnist_data, [train_size, test_size])
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Initialize the network and the optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

@track_emissions()
def train():
    # Train the network
    for epoch in range(2):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
train()
print('Finished Training')

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        net(images)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))



# generate a sine wave
sequence_length = 20
num_samples = 1000
x = np.linspace(0, num_samples * np.pi/2, num_samples)
sin_wave = np.sin(x)

# prepare data for RNN
input_data = [sin_wave[i:i+sequence_length] for i in range(sin_wave.shape[0] - sequence_length)]
target_data = [sin_wave[i+sequence_length] for i in range(sin_wave.shape[0] - sequence_length)]

input_data = torch.tensor(input_data).float().unsqueeze(2)
target_data = torch.tensor(target_data).float()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# hyperparameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.01

model = RNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
@track_emissions()
def train2():
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        outputs = model(input_data)
        loss = criterion(outputs, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch:{epoch+1}/{num_epochs}, Loss:{loss.item():.4f}')

train2();
# testing
model.eval()
test_seq = input_data[0]
predictions = []
for _ in range(input_data.shape[0]):
    with torch.no_grad():
        model_out = model(test_seq.unsqueeze(0))
        predictions.append(model_out)
        test_seq = torch.cat((test_seq[1:], model_out), 0)

plt.plot(sin_wave, label="Real Data")
plt.plot(predictions, label="Predictions")
plt.legend()
plt.show()

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
