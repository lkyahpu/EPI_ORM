# EPI_ORM

We release the code of [EPI-based Oriented Relation Networks for Light Field Depth Estimation](https://arxiv.org/abs/2007.04538).  

## Overall network

<div align=center><img src ="https://github.com/lkyahpu/EPI_ORM/raw/master/images/network.png" width="800" height="400"/></div>
<font size=2>The proposed network architecture.

## The proposed oriented relation module 

<div align=center><img src ="https://github.com/lkyahpu/EPI_ORM/raw/master/images/ORM.png" width="800" height="200"/></div>
<font size=2> The architecture of ORM. 

## Installation

[Python 3.6](https://www.anaconda.com/distribution/) 

[Anaconda 4.2.0 (64-bit)](https://www.anaconda.com/distribution/)

[Kreas 2.1.6 ](https://keras.io/)

[Matlab 2014b](https://www.mathworks.com/products/matlab.html)
```
|-- ref_aug
    |-- im_refocus.m
    |-- pinhole.m
    |-- refocus.c
```
`im_refocus.m` requires MATLAB along with a C++ compiler configured to work with MATLAB's `mex` command, the last is required for building the `refocus.mexw64` MEX function. You can check that a compiler is properly configured by executing:
```
>> mex -setup C++

>> mex refocus.c 
```
from the MATLAB command prompt.

## Data

### Dataset：4D Light Field Benchmark

We use the [4D light field benchmark](https://lightfield-analysis.uni-konstanz.de/) as our experimental dataset.
The dataset includes 24 carefully designed scenes with ground-truth disparity maps.
Each scene has 9 × 9 angular resolution and 512 × 512 spatial resolution. 16 scenes are used for training and the remaining 8 scenes for testing.

### EPIs from the light field

We extract EPIs from the light field as input. The proposed network predicts the depth of the center pixel from the pair of EPI patches.
We randomly sample the horizontal and vertical EPI patch pairs of size 9 × 29 × 3 from each scene as inputs.
<div align=center><img src="https://github.com/lkyahpu/EPI_ORM/raw/master/images/EPI.png" width="700" height="400" /></div>

### Data pre-processing

* Put the light field data into [hci_dataset/](/hci_dataset/):
```
|-- hci_dataset
    |-- additional
    |-- stratified
    |-- training
    |-- test
```

* Run `im_refocus.m` for data augmentation.


