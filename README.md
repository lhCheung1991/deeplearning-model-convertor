# Non-universal deeplearning model convertor
Although there are so many frameworks for deeplearning engineers to develop their algorithms, you can't just finish you work with one chop of the axe. For example, you may want to train your network on TensorFlow but deploy it on Caffe2. 

Maybe [ONNX](https://onnx.ai/) could be one of your axes, but it doesn't support TensorFlow now... 

This project aims to offer convertors for some specific networks, like `MobileNets from TensorFlow to Caffe2`.

Please refer to specific `README.md` for the usages.

# Support
|Network|   From|     To|   Code Folder|
|:----------------|----------------:|----------------:|----------------:|
|MobileNets|TensorFlow|Caffe2|mobilenets-tensorflow-to-caffe2|
