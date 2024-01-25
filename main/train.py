import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import sys
import os

resnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
sys.path.append(resnet_dir)

from cifar import CIFAR10

# 데이터 전처리
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# CIFAR-10 데이터 로드
train_dataset = CIFAR10(transform=transform)  
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

print(len(train_dataset))
# 모델 인스턴스화 (model.py에서 정의한 모델 사용)
from model import ResNet, Bottleneck
from model import initialize_weights
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
model.to('cuda')
net = model
net.apply(initialize_weights)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay = 0.0001)

# Training
num_epochs = 1538 #60000/469
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images = data['image'].to('cuda')  
        labels = data['label'].to('cuda')  
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 600 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
            running_loss = 0.0
        if epoch == 820:
            lr = 0.01
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 1230:
            lr = 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
torch.save(model.state_dict(), 'model_weights.pth')
print('Finished Training')