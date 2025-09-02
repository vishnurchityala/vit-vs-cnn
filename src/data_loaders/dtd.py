import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),               # Slightly larger for random crop
    transforms.RandomResizedCrop(224),           # Random crop to 224x224
    transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
    transforms.ColorJitter(brightness=0.2,       # Random brightness
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
    transforms.RandomRotation(15),               # Random rotation ±15 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation / Test transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



# Simple configuration
batch_size = 64
dtd_num_classes = 47

# Load datasets
train_dataset = datasets.DTD(root='./data', split='train', download=True, transform=train_transform)
val_dataset = datasets.DTD(root='./data', split='val', download=True, transform=val_transform)
test_dataset = datasets.DTD(root='./data', split='test', download=True, transform=val_transform)

# Simple DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Export for easy import
dtd_train_loader = train_loader
dtd_val_loader = val_loader
dtd_test_loader = test_loader