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

### EPIs from the Light Field

We extract EPIs from the light field as input. The proposed network predicts the depth of the center pixel from the pair of EPI patches.
We randomly sample the horizontal and vertical EPI patch pairs of size 9 × 29 × 3 from each scene as inputs.
<div align=center><img src="https://github.com/lkyahpu/EPI_ORM/raw/master/images/EPI.png" width="700" height="400" /></div>

### Data Augmentation

* Put the light field data into [hci_dataset/](/hci_dataset/):

You need to download HCI Light field dataset from http://hci-lightfield.iwr.uni-heidelberg.de/.
Unzip the LF dataset and move `additional/, training/, test/, stratified/ ` into the `hci_dataset/`.

```
|-- hci_dataset
    |-- additional
    |-- stratified
    |-- training
    |-- test
```

* Run `im_refocus.m` for data augmentation. Data will be saved `ref_aug/output_ref/XX/XX_ref_alpha.png` (XX is the scene, `alpha` is the disparity shift). 
* You might be change the setting line 96 `index=1:10` to change the number of refocusing.

## train

* Run `python epi_train.py` to train our model, you need to modify the line 208 like below 
`dir_LFimages  = ['hci_dataset/additional/'+LFimage for LFimage in os.listdir('/hci_dataset/additional') if LFimage != 'license.txt']`

* Run `save_load_loss.py` to plot the loss curve.

## Test

* The pretrained models are provided in the repo. 

* Run `python EPI_test/test_LF.py` to test and save the test results, you need to modify the line 35 like below 
`valdata_fill_x ,valdata_fill_y ,numy,numx= read_pinhole_LF(9,9,'E:/LKY/dataset/benchmark/cotton')`

* Run `python func_epimodel.py` to obtain the network information.

## Results

Qualitative results on the 4D light field benchmark [7]. For each scene, the top row shows the estimated disparity maps and the bottom row shows the error maps for BadPix.
(a) Ground truth. (b) LF [1]. (c) CAE [2]. (d) LF_OCC [3]. (e) SPO [4]. (f) EPN [5]. (g)EPINET [6]. (h) Ours.
<div align=center><img src ="https://github.com/lkyahpu/EPI_ORM/raw/master/images/qualitative-results.png" width="700" height="800"/></div>
<font size=2> Visual comparison of ours and other state-of-the-art methods on the 4D light field benchmark.
    
## References
1. H. Jeon, J. Park, G. Choe, J. Park, Y. Bok, Y. Tai, and I. S. Kweon. Accurate depth map estimation from a lenslet light field camera. In Proceedings of IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pages 1547–1555, 2015.
2. W. Williem and I. K. Park. Robust light field depth estimation for noisy scene with occlusion. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pages 4396–4404, 2016.
3. T.-C. Wang, A. Efros, and R. Ramamoorthi. Occlusion-aware depth estimation using light-field cameras. In Proceedings of International Conference on Computer Vision
(ICCV), pages 3487–3495, 2015.
4. S. Zhang, H. Sheng, C. Li, J. Zhang, and Z. Xiong. Robust depth estimation for light field via spinning parallelogram operator. Computer Vision and Image Understanding
(CVIU), 145:148–159, 2016.
5. Y. Luo, W. Zhou, J. Fang, L. Liang, H. Zhang, and G. Dai. Epi-patch based convolutional neural network for depth estimation on 4d light field. In International Conference
on Neural Information Processing, pages 642–652, 2017.
6. C. Shin, H. Jeon, Y. Yoon, I. S. Kweon, and S. J. Kim. Epinet: A fully-convolutional neural network using epipolar geometry for depth from light field images. In Proceedings
of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4748–4757, 2018.

## TODO
Training code release  

## Contact

[Kunyuan Li](mailto:lkyhfut@gmail.com),  [Jun Zhang](mailto:zhangjun1126@gmail.com)

Questions can also be left as issues in the repository. We will be happy to answer them.
