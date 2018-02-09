from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from functools import wraps

from models import mobilenet_v1


def mobilenet_factory(depth_multiplier=1.0, default_image_size=224, scope="MobilenetV1"):
    mobilenet_v1.default_image_size = default_image_size
    @wraps(mobilenet_v1.mobilenet_v1)
    def mobilenet(inputs, num_classes, is_training):
        return mobilenet_v1.mobilenet_v1(inputs=inputs, num_classes=num_classes, is_training=is_training, depth_multiplier=depth_multiplier, scope=scope) 
    return mobilenet

mobilenet_zoo = {"mobilenet_v1_100_224": mobilenet_factory(depth_multiplier=1.0, default_image_size=224, scope='MobilenetV1'),
                 "mobilenet_v1_100_192": mobilenet_factory(depth_multiplier=1.0, default_image_size=192, scope='MobilenetV1'),
                 "mobilenet_v1_100_160": mobilenet_factory(depth_multiplier=1.0, default_image_size=160, scope='MobilenetV1'),
                 "mobilenet_v1_100_128": mobilenet_factory(depth_multiplier=1.0, default_image_size=128, scope='MobilenetV1'),
                 "mobilenet_v1_75_224": mobilenet_factory(depth_multiplier=0.75, default_image_size=224, scope='MobilenetV1'),
                 "mobilenet_v1_75_192": mobilenet_factory(depth_multiplier=0.75, default_image_size=192, scope='MobilenetV1'),
                 "mobilenet_v1_75_160": mobilenet_factory(depth_multiplier=0.75, default_image_size=160, scope='MobilenetV1'),
                 "mobilenet_v1_75_128": mobilenet_factory(depth_multiplier=0.75, default_image_size=128, scope='MobilenetV1'),
                 "mobilenet_v1_50_224": mobilenet_factory(depth_multiplier=0.50, default_image_size=224, scope='MobilenetV1'),
                 "mobilenet_v1_50_192": mobilenet_factory(depth_multiplier=0.50, default_image_size=192, scope='MobilenetV1'),
                 "mobilenet_v1_50_160": mobilenet_factory(depth_multiplier=0.50, default_image_size=160, scope='MobilenetV1'),
                 "mobilenet_v1_50_128": mobilenet_factory(depth_multiplier=0.50, default_image_size=128, scope='MobilenetV1'),
                 "mobilenet_v1_25_224": mobilenet_factory(depth_multiplier=0.25, default_image_size=224, scope='MobilenetV1'),
                 "mobilenet_v1_25_192": mobilenet_factory(depth_multiplier=0.25, default_image_size=192, scope='MobilenetV1'),
                 "mobilenet_v1_25_160": mobilenet_factory(depth_multiplier=0.25, default_image_size=160, scope='MobilenetV1'),
                 "mobilenet_v1_25_128": mobilenet_factory(depth_multiplier=0.25, default_image_size=128, scope='MobilenetV1')
                    }
