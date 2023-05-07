"""
https://github.com/apple/ml-mobileone/blob/main/mobileone.py
pytorch -> tensorflow
"""
from typing import List

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input, Sequential


def get_tensor_shape(tensor):
    B = tf.shape(tensor)[0]
    H = tf.shape(tensor)[1]
    W = tf.shape(tensor)[2]
    C = tf.shape(tensor)[3]
    return B, H, W, C


def se_block(
    inputs,
    filters: int,
    rd_ratio: float = 0.0625
):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """
    # C = get_tensor_shape(inputs)[-1]
    x = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    x = layers.Conv2D(
        int(filters * rd_ratio),
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=True
    )(x)  # reduction
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=True
    )(x)  # expand
    x = layers.Activation('sigmoid')(x)
    x = layers.Multiply()([inputs, x])
    return x


def mobileone_block(
    inputs,
    filters: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    inference_mode: bool = False,
    use_se: bool = False,
    num_conv_branches: int = 1
):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    # Multi-branched train-time forward pass.
    # Skip branch output
    if (inputs.shape[-1] == filters) and (stride == 1):
        identity_out = layers.BatchNormalization()(inputs)
    else:
        identity_out = 0

    # Scale branch output
    if kernel_size > 1:
        scale_out = conv_bn(inputs, filters, 1, stride, padding, groups)
    else:
        scale_out = 0

    # Other branches
    out = identity_out + scale_out
    for _ in range(num_conv_branches):
        out += conv_bn(inputs, filters, kernel_size, stride, padding, groups)

    if use_se:
        out = se_block(out, filters, 0.0625)
    out = layers.Activation('relu')(out)
    return out


def conv_bn(
    inputs,
    filters: int,
    kernel_size: int,
    strides: int,
    padding: str,
    groups: int,
    use_bias: bool = False
):
    return Sequential(
        [
            layers.ZeroPadding2D(padding),
            layers.Conv2D(
                filters,
                kernel_size,
                strides,
                'valid',
                groups=groups,
                use_bias=use_bias
            ),
            layers.BatchNormalization(),
        ]
    )(inputs)


def get_kernel_bias() -> List[tf.Tensor, tf.Tensor]:
    """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
    """
    pass


def fuse_bn_tensor(branch) -> List[tf.Tensor, tf.Tensor]:
    """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
    """
    if isinstance(branch, Sequential):
        conv = branch.get_layer(index=1)  # conv
        bn = branch.get_layer(index=2)  # bn

        kernel = conv.get_weights()[0]

        running_mean = bn.moving_mean
        running_var = bn.moving_variance
        gamma = bn.gamma
        beta = bn.beta
        eps = bn.epsilon
    else:
        assert isinstance(branch, layers.BatchNormalization)


def reparameterize_block():
    pass


def MobileOne():
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    pass
