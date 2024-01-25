import torch
import pickle
import cv2
from torchvision import transforms

class CIFAR10Test(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.images = []
        self.labels = []

        with open('../data/cifar-10-batches-py/test_batch', 'rb') as f:
            db = pickle.load(f, encoding='bytes')

        images = db[b'data']
        labels = db[b'labels']

        for raw_image, label in zip(images, labels):
            image = raw_image.reshape(3, 32, 32).transpose(1, 2, 0)
            self.images.append(image)
            self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        
        label = self.labels[idx]

        inputs = {'image': image}
        targets = {'label': label}

        return inputs, targets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_dataset = CIFAR10Test(transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)