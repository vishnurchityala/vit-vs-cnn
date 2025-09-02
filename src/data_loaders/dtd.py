import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Simple transforms for DTD dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Simple configuration
batch_size = 64
dtd_num_classes = 47

# Load datasets
train_dataset = datasets.DTD(root='./data', split='train', download=True, transform=transform)
val_dataset = datasets.DTD(root='./data', split='val', download=True, transform=transform)
test_dataset = datasets.DTD(root='./data', split='test', download=True, transform=transform)

# Simple DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Export for easy import
dtd_train_loader = train_loader
dtd_val_loader = val_loader
dtd_test_loader = test_loader