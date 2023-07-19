import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from codecarbon import EmissionsTracker
from torch.profiler import profile, record_function, ProfilerActivity

# Check if a GPU is available and if not, default to CPU
device = torch.device("cuda")

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        linear_output = self.linear(x)
        encoded = self.encoder(linear_output.unsqueeze(0))  # Add an extra dimension for seq_len
        output = self.fc(encoded.squeeze(0))  # Remove the extra dimension after encoding
        return output

# Hyperparameters
batch_size = 64
input_dim = 784  # Input dimension is 28x28 for MNIST, so flattened size is 784
output_dim = 10  # Number of classes
hidden_dim = 256  # Hidden dimension of the model
num_layers = 4  # Number of transformer layers
num_heads = 8  # Number of attention heads
learning_rate = 0.001
num_epochs = 2

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model initialization
model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads).to(device)  # Move model to device
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# CodeCarbon emission tracking
tracker = EmissionsTracker(project_name="My Project")
tracker.start()

# Training loop with Profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)  # Move data and target to device
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

# Evaluation loop
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.view(data.size(0), -1).to(device), target.to(device)  # Move data and target to device
        output = model(data)
        _, predicted = torch.max(output, dim=1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)

accuracy = total_correct / total_samples
print(f"Accuracy: {accuracy * 100}%")

# Stop emission tracking and print emissions summary
emissions = tracker.stop()
print(emissions)
