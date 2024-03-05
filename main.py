from document_orientation_dataset import DocumentOrientationDataset
from simple_cnn import SimpleCNN
import torch
from torch import *
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define transformations (resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Example resize, adjust as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = DocumentOrientationDataset("data", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Training loop (simplified for brevity)
num_epochs = 10  # Adjust as necessary
for epoch in range(0, num_epochs):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
