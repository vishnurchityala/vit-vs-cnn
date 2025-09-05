if __name__ == "__main__":
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    from src.model import ConvNeXtXXL_MLP

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # DTD test dataset
    test_dataset = datasets.DTD(
        root="./data",
        split="test",
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # num_workers=0 on macOS

    # Load model
    model = ConvNeXtXXL_MLP(pretrained=False)
    model.load_state_dict(torch.load("./saved_models/convnext_dtd_acc_80.59.pth", map_location=device))
    model.to(device)
    model.eval()

    # Evaluate only first 20 images
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy on first 20 test images: {accuracy:.2f}%")
