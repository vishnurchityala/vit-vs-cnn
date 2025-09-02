import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 64
dtd_num_classes = 47

# Platform-aware worker configuration
# Use fewer workers on Mac for development, more on server for training
def get_num_workers():
    """Get optimal number of workers based on platform and available CPUs"""
    try:
        # Detect if we're likely on a server (Linux) or development machine (Mac/Windows)
        max_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 for safety
        
        # Use fewer workers on Mac for development
        if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
            return min(2, max_workers)  # Mac development
        else:
            return max_workers  # Server/Windows
    except Exception:
        # Fallback to safe default
        return 2

num_workers = get_num_workers()

train_dataset = datasets.DTD(root='./data', split='train', download=True, transform=transform)
val_dataset   = datasets.DTD(root='./data', split='val',   download=True, transform=transform)
test_dataset  = datasets.DTD(root='./data', split='test',  download=True, transform=transform)

# Platform-aware DataLoaders with graceful worker handling
try:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
except Exception as e:
    print(f"[WARNING] Failed to create DataLoaders with {num_workers} workers: {e}")
    print("[INFO] Falling back to single-threaded DataLoaders")
    # Fallback to simple configuration
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
