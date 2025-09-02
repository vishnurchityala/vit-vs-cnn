import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import multiprocessing
import os

transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

batch_size = 64
val_split = 0.1
cifar_num_classes = 10

# Platform-aware worker configuration
def get_num_workers():
    """Get optimal number of workers based on platform and available CPUs"""
    try:
        max_workers = min(multiprocessing.cpu_count(), 8)
        if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
            return min(2, max_workers)  # Mac development
        else:
            return max_workers  # Server/Windows
    except Exception:
        return 2

num_workers = get_num_workers()


full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

val_size = int(len(full_train_dataset) * val_split)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Platform-aware DataLoaders
try:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
except Exception as e:
    print(f"[WARNING] Failed to create CIFAR DataLoaders with {num_workers} workers: {e}")
    print("[INFO] Falling back to single-threaded DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
