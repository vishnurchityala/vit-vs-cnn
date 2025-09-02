import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(216),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 64

train_dataset = datasets.DTD(root='./data', split='train', download=True, transform=transform)
val_dataset   = datasets.DTD(root='./data', split='val',   download=True, transform=transform)
test_dataset  = datasets.DTD(root='./data', split='test',  download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=3)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=3)
