# EPI_ORM

We release the code of [EPI-based Oriented Relation Networks for Light Field Depth Estimation](https://arxiv.org/abs/2007.04538).  

## Overall network

<div align=center><img src ="https://github.com/lkyahpu/EPI_ORM/raw/master/images/network.png" width="800" height="400"/></div>
<font size=2>The proposed network architecture.

## The proposed oriented relation module 

<div align=center><img src ="https://github.com/lkyahpu/EPI_ORM/raw/master/images/ORM.png" width="800" height="200"/></div>
<font size=2> The architecture of ORM. 

## Installation

[Python 2.7](https://www.anaconda.com/distribution/) 

[Caffe](https://caffe.berkeleyvision.org/)

[Matlab 2016b](https://www.mathworks.com/products/matlab.html)

Do the following two steps to make sure that .m files can be used in Python:
- `cd matlabroot/extern/engines/python` 
- `python setup.py`

Note: matlabroot is the root directory of MATLAB in your system.

## Data

### Dataset：4D Light Field Benchmark

We use the [4D light field benchmark](https://lightfield-analysis.uni-konstanz.de/) as our experimental dataset.
The dataset includes 24 carefully designed scenes with ground-truth disparity maps.
Each scene has 9 × 9 angular resolution and 512 × 512 spatial resolution. 16 scenes are used for training and the remaining 8 scenes for testing.

### EPIs from the light field

We extract EPIs from the light field as input. The proposed network predicts the depth of the center pixel from the pair of EPI patches.
We randomly sample the horizontal and vertical EPI patch pairs of size 9 × 29 × 3 from each scene as inputs.
<div align=center><img src="https://github.com/lkyahpu/EPI_ORM/raw/master/images/EPI.png" width="700" height="400" /></div>
