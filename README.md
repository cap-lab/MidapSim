Test the simulator after setting the __C.SYSTEM.ROOT parameter in config.py

How to simulate?

0. network preparation

0.1) install requirements w/ pip install -r requirements.txt


0.2)
download the caffe2 model with
python -m caffe2.python.models.download [model_name]
or translate from the caffe model

script: ./scripts/download_caffe2_networks.sh

current supported networks (caffe2)

- bvlc_googlenet
- mobilenet
- resnet
- vgg
- squeezenet

0.3)
tensorflow model support
you can simulate built-in networks w/ builtin option

mobilenet_v2
resnet50
inceptionV3

or simulate with download hdf5 files (keras)

0.4)
-v1.3.0)
From now on, you can define your own network with ModelBuilder
please refer test code with models/model_builder.py and models/examples.py 

1. modify config.py for the MIDAP configuration

please specify model generation option & hardware option

2. you can use TestWrapper class for easy simulation.

3. please refer test.py and test_wrapper.py
