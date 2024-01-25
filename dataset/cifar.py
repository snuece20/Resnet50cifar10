## TODO
# train/test control flow
# augmentation: flip, crop ....


import torch
import pickle
import cv2
from PIL import Image

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.images = []
        self.labels = []
        
        for i in range(1,6):
            with open(f'../data/cifar-10-batches-py/data_batch_{i}', 'rb') as f:
                db = pickle.load(f, encoding='bytes')
            
            images = db[b'data']
            labels = db[b'labels']

            for raw_image, label in zip(images, labels):
                image = raw_image.reshape(3, 32, 32).transpose(1,2,0)
            

                # cv2.imwrite('tmp.png', image[:,:,::-1])
                # import pdb;pdb.set_trace()

                self.images.append(image)
                self.labels.append(label)

        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        
        label = self.labels[idx]

        return {'image': image, 'label': label}

if __name__ == "__main__":
    from torchvision import transforms

    dataset = CIFAR10(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]))

    _dataset = iter(dataset)
    inputs, targets = next(_dataset)

    #import pdb;pdb.set_trace()
#