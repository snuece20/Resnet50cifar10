# ResNet50 implementing with cifar10 data
  
## Introduction
This repository is Pytorch implenmentation of ResNet50, based on paper "arXiv:1512.03385".
  
## Directory  
### ResNet50p  
  
The ${ROOT} is described as below.  
  
```
${ROOT}  
|-- data  
|-- dataset  
|-- main  
```
  
* data contains cifa10 batches (test batch and batches for training) which loaded from datasetcodes.  
* dataset contains data loading codes.  
* main contains codes for training and testing cifar10 data with ResNet50 network. output data(trained weights and test result) also saved in main directory.  
   
### Data   
The data directory is configured as shown in the structure below.  
  
```
${ROOT}  
|-- data  
|   |-- cifar-10-batches-py  
|   |   |-- batches.meta  
|   |   |-- data_batch_1  
|   |   |-- data_batch_2  
|   |   |-- data_batch_3  
|   |   |-- data_batch_4  
|   |   |-- data_batch_5  
|   |   |-- test_batch  
|   |-- cifar-10-python.tar.gz  
```
  
* If you run the cifar.py and cifar_test.py files in the dataset directory, the above structure is automatically formed.  

### Dataset  
  
```
${ROOT}  
|-- dataset  
|   |-- __pycache__  
|   |-- cifar_test.py  
|   |-- cifar.py  
```
  

* You can load cifar10 datas for training and testing by implementing cifar.py and cifar_test.py files.  

### Main  

```
${ROOT}  
|-- main  
|   |-- __pycache__  
|   |-- __init__.py  
|   |-- model_weights.pth  
|   |-- model.py  
|   |-- test.py  
|   |-- train.py  
```
  
 
* If you run the train.py file to train, the trained weights are stored in the model_weights.pth file.  
* __init__.py file causes the directory to be considered a package.  
    
## Running ResNet50  
### Start  
* Install PyTorch and Python.  
Activate torch by below code.
```
conda activate torch
```
  
### Train  
Run below code to train ResNet50.  
```python
python train.py  
```
  
* If implementing is finished the output will be "Finished Training" and the trained weights are stored in the model_weights.pth file.  

### Test  
Run below code to test outcome of ResNet50 model trained by cifar10 data.  
```python
python test.py  
```

## Results  
Results ouput of testing should be "Accuracy of the network on the test images : 94.98%"  


* The results are more accurate than those presented in the paper.
  
<p align="center"><img width="401" alt="image" src="https://github.com/snuece20/Resnet50cifar10/assets/157671957/4908fb4f-377f-470f-a411-0bd0f953ca92"></p>  
<p align="center"> Image from "arXiv:1512.03385" </p>  


* The paper doesn't provide the number of datapoints used for normalization, so our code uses a normalization method optimized for CIFAR10. It is assumed that this is the reason for the difference.

## Reference  
```
@InProceedings{IEEE_2015_CVPR,  
author = {Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun},  
title = {Deep Residual Learning for Image Recognition},  
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},  
year = {2015}  
}
```


 


