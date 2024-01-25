import torch
from torchvision import transforms
import sys
import os

resnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
sys.path.append(resnet_dir)

from cifar_test import CIFAR10Test, test_loader
from model import ResNet, Bottleneck
from model import ResNet50
   



# ResNet50 인스턴스를 생성하고 평가 모드로 설정
model = ResNet50()
model.to('cuda') 
model.eval()
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
# 정확도 계산을 위한 변수 초기화
correct = 0
total = 0

# 평가 수행
with torch.no_grad():
    for inputs, targets in test_loader:
        images, labels = inputs['image'].to('cuda'), targets['label'].to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
