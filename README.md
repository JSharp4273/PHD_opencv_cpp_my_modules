# PHD_opencv_cpp_my_modules
This GitHub repository contains several folders containing algorithms that I implemented during my Ph.D. The purpose of each folder is as follows:

-"xcore": This folder contains essential functions for managing input and hardware accelerated functions for working with Meta-Templates. It also includes tools for implementing more efficient functions using the TAPI and high-priority parallel for loop. Additionally, a linear algebra function for solving least-squares problems and other functions like Kronecker product can be found in this folder. All the functions in this folder benefit from the TAPI.

-"xquality": This folder contains the implementation of three metrics: Mean Absolute Error, GMLOG-BIQA, and an SVD-based metric. The GMLOG-BIQA was proposed by Xue et al. in the paper "Blind image quality assessment using joint statistics of gradient magnitude and Laplacian features" published in IEEE Transactions on Image Processing, 2014. The SVDBased metric was introduced by Shnayderman et al. in the paper "An SVD-Based Grayscale Image Quality Measure for Local and Global Assessment" published in IEEE Transactions on Image Processing, 2006. Both GMLOG-BIQA and the SVDBased metric have publicly available MATLAB code. Please note that GMLOG-BIQA code does not have any license, so NO LICENSE has been given to this folder. However, the Mean Absolute Error code is under Apache 2.0 license and the SVDBased metric is under the original BSD-2 clause license.

-"mlx" (ml extra): This folder adds multidimensional Linear Regression and Partial Least Square Regression (PLSR) algorithms. The PLSR algorithm is based on the SIMPLS algorithm introduced by De Jong in the paper "SIMPLS: an alternative approach to partial least squares regression" published in Chemometrics and Intelligent Laboratory Systems, 1993. Both algorithms can benefit from the TAPI for training and prediction.

-"cudaxcore": This folder offers the same arithmetic functions as the "xcore" folder, along with an interface with the arrayFire library, which implements several standard linear algebra decomposition algorithms such as the SVD. Other functions in this module are currently under revision.

-"cudaml": This folder implements the same algorithms as the "mlx" folder, but using the OpenCV's CUDA API for GPU acceleration.

Please note that these functions were developed at different moments during the last few years. I have compiled a custom OpenCV-4.5.3 with CUDA 11.4 and successfully tested all of them. These codes were evaluated under Ubuntu 20.04 in September 2021. Also note that the most of the code is licenced under BSD-3 clause, some algorithms (e.g. GMLOG-BIQA) are NOT licenced.
