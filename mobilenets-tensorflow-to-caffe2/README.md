### Start jupyter notebook
```shell
cd mobilenets-tensorflow-to-caffe2
ipython notebook --ip="*" --port=8391 --notebook-dir=.
```

### Launch notebook
```
# launch converting tool and run it, it will load mobilenet model of TensorFlow 
# from `examples/mobilenets-tensorflow-ckpt` and generate model for Caffe2
convert_mobilenets_tensorflow_to_caffe2.ipynb

# verify the result
test_convert_result.ipynb
```

### 中文开发者
请参考 [TensorFlow 到 Caffe2 的 MobileNets 转换器](https://lhcheung1991.github.io/blogs/2017/08/24/convert-mobilenet-from-tensorflow-to-caffe2.html)
