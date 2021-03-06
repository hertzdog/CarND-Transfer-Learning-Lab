# Transfer Learning Lab with VGG, Inception and ResNet
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lab, you will continue exploring transfer learning. You've already explored feature extraction with AlexNet and TensorFlow. Next, you will use Keras to explore feature extraction with the VGG, Inception and ResNet architectures. The models you will use were trained for days or weeks on the [ImageNet dataset](http://www.image-net.org/). Thus, the weights encapsulate higher-level features learned from training on thousands of classes.

We'll use two datasets in this lab:

1. German Traffic Sign Dataset
2. Cifar10

Unless you have a powerful GPU, running feature extraction on these models will take a significant amount of time. To make things we precomputed **bottleneck features** for each (network, dataset) pair, this will allow you experiment with feature extraction even on a modest CPU. You can think of bottleneck features as feature extraction but with caching.  Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed once through the network we can cache and reuse the output.

The files are encoded as such:

- {network}_{dataset}_bottleneck_features_train.p
- {network}_{dataset}_bottleneck_features_validation.p

network can be one of 'vgg', 'inception', or 'resnet'

dataset can be on of 'cifar10' or 'traffic'

How will the pretrained model perform on the new datasets?

** CIFAR **
Netowrk used:

    # 1st Layer - Convolutional with 32 filters, a 3x3 kernel, and valid padding before the flatten layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32,32,3)))

    # 2nd Layer - Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))

    # 3rd Layer - Dropout
    model.add(Dropout(dropout))

    # 4th layer - ReLU activation layer
    model.add(Activation('relu'))

    # 5th Layer - Add a flatten layer
    model.add(Flatten(input_shape=(32, 32, 3)))

    # 6th Layer - Add a fully connected layer
    model.add(Dense(128))

    # 7th Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    # 8th Layer - 43 classes - 43 neurons - Output or 10 for cifar10
    model.add(Dense(10))

    # 9th layer softmax
    model.add(Activation('softmax'))


Started with it, we got, after 10 epochs:
Epoch 10/10
40000/40000 [==============================] - 8s - loss: 0.6755 - acc: 0.7589 - val_loss: 0.9912 - val_acc: 0.6685



Here are the links for download:
https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip
https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip
https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip
