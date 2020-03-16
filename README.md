# Decsription
This work is used the model of mtcnn which trained by pytorch and processed by C++ to realize a face detect.
# Prerequisites
You need to know loading a  torchScript model in C++ ,you can learn from the website of [pytorch](https://pytorch.org/tutorials/advanced/cpp_export.html). You also need to know how to use opencv.
# Dependencies
C++ <br>
opencv <br>
libtorch <br>
Ubuntu/windows <br>
Cuda 10 <br>
# Ubuntu
If you want to test on Ubuntu. Just Install all the dependencies and use make to build the project. 
# windows
If you want to test on windows. You need to install cmake and vs2015/2017 .<br> 
create a directory named build .<br>
The command of cmake is : cmake -DCMAKE_PREFIX_PATH=D:/opencv/build/x64/vc14/lib;D:/libtorch -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 Win64 " ..<br>
