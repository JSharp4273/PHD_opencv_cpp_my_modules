# PHD_opencv_cpp_my_modules
These folders contain several algorithms I implemented during my Ph.D.

The folder "xcore" contains essential functions to manage input and hardware accelerated functions for working with Meta-Templates.
It also contains some tools to implement more efficient functions using the TAPI and high priority parallel for loop.
A linear algebra function is also added to solve least-squares problems.
Some other functions, such as the Kronecker product, can be found in this folder.
Note that all the functions benefit from the TAPI.

The folder "xquality" contains the implementation of three metrics, the Mean Absolute Error, GMLOG-BIQA, and an SVD based metric.
The GMLOG-BIQA was proposed by Xue et al : Blind image quality assessment using joint statistics of gradient magnitude and laplacian features.IEEE Transactions on ImageProcessing, 23(11):4850–4862, 2014.
The SVDBased metric was introduced by Shnayderman et al. : An SVD-Based Grayscale Image Quality Measure for Local and Global Assessment",  IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 15, NO. 2, FEBRUARY 2006.
Both GMLOG-BIQA and the SVDBased metric have availlable MatLab code publicly.
Because GMLOG-BIQA code does not have any license, NO LICENSE has been given to this folder.
Nonetheless, the Mean Absolute Error code is under Apache 2.0 license and the SVDBased metric (with respect to the original BSD-2 clause license).

The folder "mlx" (ml extra) adds the multidimensional Linear Regression, as well as the Partial Least Square Regression.
Like "xcore" both algorithms can benefit from the TAPI.

The folder "cudaxcore" offers the same arithmetics functions, whose can be found in the  "xcore" folder.
It also offers an interface with the library arrayFire, which implements several standard linear algebra decomposition algorithms such as the SVD.
Other functions for this module are currently under revision.

The folder "cudaml" implements the same algorithm that can be found in the folder "mlx", using the OpenCV"s CUDA API.

Note that these functions were initially developed at different moments during the last few years.
I have compiled a custom OpenCV-4.5.3 with CUDA 11.4, and test successfully tested all of them.
These codes were evaluated under UBUNTU 20.04 in September 2021.
