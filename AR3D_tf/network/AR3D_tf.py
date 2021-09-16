import sys
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
sys.path.append('..')
sys.path.append('/home/subin/AR3D/AR3D/')
from config_tf import config


# from config import Path
def Pool3d(kernel_size, stride):
    return tf.keras.layers.MaxPool3D(kernel_size, stride)


def Conv3d(out_channel, kernel_size, activation):
    return tf.keras.layers.Conv3D(out_channel, kernel_size, padding='same', activation=activation)


def Softmax():
    return tf.keras.layers.Softmax()


def ReLU():
    return tf.keras.layers.ReLU()


def FC(out_dim):
    return tf.keras.layers.Dense(out_dim)


def BatchnNorm():
    return tf.keras.layers.BatchNormalization()


def DropOut(rate):
    return tf.keras.layers.Dropout(rate)


class sfe_block(tf.keras.layers.Layer):
    def __init__(self):
        super(sfe_block, self).__init__()
        self.conv1 = Conv3d(64, (3, 3, 3), 'relu')
        self.pool1 = Pool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = Conv3d(128, (3, 3, 3), 'relu')
        self.pool2 = Pool3d((2, 2, 2), (2, 2, 2))
        self.conv3_a = Conv3d(256, (3, 3, 3), 'relu')
        self.conv3_b = Conv3d(256, (3, 3, 3), 'relu')
        self.pool3 = Pool3d((2, 2, 2), (2, 2, 2))
        self.conv4_a = Conv3d(512, (3, 3, 3), 'relu')
        self.conv4_b = Conv3d(512, (3, 3, 3), 'relu')
        self.pool4 = Pool3d((2, 2, 2), (2, 2, 2))

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3_a(x)
        x = self.conv3_b(x)
        x = self.pool3(x)
        x = self.conv4_a(x)
        x = self.conv4_b(x)
        out = self.pool4(x)

        return out


class r3d_module(tf.keras.layers.Layer):
    def __init__(self):
        super(r3d_module, self).__init__()
        self.r_conv1 = Conv3d(128, (1, 1, 1), 'relu')
        self.r_conv1_bn = BatchnNorm()
        self.r_conv2 = Conv3d(128, (1, 3, 3), 'relu')
        self.r_conv2_bn = BatchnNorm()
        self.r_conv3 = Conv3d(128, (3, 1, 1), 'relu')
        self.r_conv3_bn = BatchnNorm()
        self.r_conv4 = Conv3d(512, (1, 1, 1), None)
        self.r_conv4_bn = BatchnNorm()

    def call(self, x):
        x = self.r_conv1(x)
        x = self.r_conv1_bn(x)
        x = self.r_conv2(x)
        x = self.r_conv2_bn(x)
        x = self.r_conv3(x)
        x = self.r_conv3_bn(x)
        x = self.r_conv4(x)
        out = self.r_conv4_bn(x)

        return out


class a3d_module(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio, attention_method):
        super(a3d_module, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.reduction_channel = int(512 / self.reduction_ratio)
        self.k_conv = Conv3d(self.reduction_channel, (1, 1, 1), None)
        self.q_conv = Conv3d(self.reduction_channel, (1, 1, 1), None)
        self.v_conv = Conv3d(self.reduction_channel, (1, 1, 1), None)
        self.restore_conv = Conv3d(512, (1, 1, 1), None)

        self.softmax = Softmax()

    def call(self, x):
        shape = [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], self.reduction_channel]  ## feature_size x channel
        # shape = [config.batch_size , 98, self.reduction_channel] ## feature_size x channel

        k = self.k_conv(x)
        k = tf.reshape(k, shape)
        q = self.q_conv(x)
        q = tf.reshape(q, shape)

        w_spatio_temporal = tf.einsum('b i c, b j c -> b i j', k, q)
        # w_channel = tf.einsum('b f i, b f j -> b i j', k, q)

        w_spatio_temporal = self.softmax(w_spatio_temporal / np.sqrt(x.shape[2] * x.shape[3] * x.shape[4]))
        # w_channel = self.softmax(w_channel)

        v = self.v_conv(x)
        v = tf.reshape(v, shape)

        # spatio_temporal
        out = tf.einsum('b i f, b f j -> b i j', w_spatio_temporal, v)
        # out = tf.einsum('b c i, b j c -> b i j', w_channel, v) ## ?? recheck this einsum op
        out = tf.reshape(out, [x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.reduction_channel])
        # out = tf.reshape(out,[config.batch_size, 2, 7, 7, self.reduction_channel])
        out = self.restore_conv(out)

        return out


class AR3D(tf.keras.Model):
    def __init__(self, num_classes, AR3D_V='v2', SFE_type='t1', attention_method='spatio_temporal', reduction_ratio=4,
                 hidden_unit=4096):
        super(AR3D, self).__init__()
        ## sfe
        if SFE_type == 't1':
            self.sfe = sfe_block()
            ## fc_input_size = 50176
        else:
            pass
            # ## SFE_type == 't2'
            # ## sfe layer from c3d's convolution layer
            # self.sfe = c3d_conv()
            ## fc_input_size = 8192

        ## dfe
        self.version = AR3D_V
        self.residual = r3d_module()
        self.attention = a3d_module(reduction_ratio, attention_method)

        ## prediction (fc)
        # #######
        # self.pool = Pool3d((2,2,2),(2,2,2))
        # #######
        self.fc6 = FC(hidden_unit)
        self.fc7 = FC(4096)
        self.fc8 = FC(num_classes)

        self.relu = ReLU()
        self.dropout = DropOut(0.5)

    def call(self, x):
        ## sfe
        x = self.sfe(x)

        ## dfe
        if self.version == 'v1':
            ## AR3D_V == 'v1'
            # x = self.relu(self.residual(x) + self.attention(x) + x)

            sc = x
            residual = self.residual(x)
            attention = self.attention(x)
            x = self.relu(attention + residual + sc)
        else:
            ## AR3D_V == 'v2' (default)
            # x = self.residual(x) + x
            # x = self.relu(x)
            # x = self.attention(x) + x
            sc_r3d = x
            x = self.residual(x)
            x = self.relu(x + sc_r3d)

            sc_a3d = x
            x = self.attention(x)
            x = x + sc_a3d

        ## prediction
        ######
        # x = self.pool(x)
        ######
        x = tf.reshape(x, [x.shape[0], -1])
        x = self.fc6(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.dropout(x)
        x = self.fc8(x)

        return x

# class AR3D_sequential(tf.keras.Model):
#   def __init__(self):
#     super(AR3D_sequential, self).__init__()
#     # sfe block
#     self.conv1 = Conv3d_relu(64,(3,3,3))
#     self.pool1 = Pool3d((1,2,2),(1,2,2))
#     self.conv2 = Conv3d_relu(128,(3,3,3))
#     self.pool2 = Pool3d((2,2,2),(2,2,2))
#     self.conv3_a = Conv3d_relu(256,(3,3,3))
#     self.conv3_b = Conv3d_relu(256,(3,3,3))
#     self.pool3 = Pool3d((2,2,2),(2,2,2))
#     self.conv4_a = Conv3d_relu(512,(3,3,3))
#     self.conv4_b = Conv3d_relu(512,(3,3,3))
#     self.pool4 = Pool3d((2,2,2),(2,2,2))

#     # r3d
#     self.r_conv1 = Conv3D_nonact(128,(1,1,1))
#     self.r_conv1_bn = BatchnNorm()
#     self.r_conv2 = Conv3D_nonact(128,(1,3,3))
#     self.r_conv2_bn = BatchnNorm()
#     self.r_conv3 = Conv3D_nonact(128,(3,1,1))
#     self.r_conv3_bn = BatchnNorm()
#     self.r_conv4 = Conv3D_nonact(512,(1,1,1))
#     self.r_conv4_bn = BatchnNorm()

#     # attention

#     self.reduction_ratio = 4
#     self.reduction_channel = 512 / self.reduction_ratio
#     self.k_conv = Conv3D_nonact(self.reduction_channel,(1,1,1))
#     self.q_conv = Conv3D_nonact(self.reduction_channel,(1,1,1))
#     self.v_conv = Conv3D_nonact(self.reduction_channel,(1,1,1))
#     self.restore_conv = Conv3D_nonact(512,(1,1,1))

#     self.fc6 = FC(4096)
#     self.fc7 = FC(4096)
#     self.fc8 = FC(101)
#     # 101 == numclasses

#     self.softmax = Softmax()
#     self.relu = ReLU()
#     self.dropout = DropOut(0.5)

#   def call(self, x):
#     # sfe block
#     x = self.conv1(x)
#     x = self.pool1(x)
#     x = self.conv2(x)
#     x = self.pool2(x)
#     x = self.conv3_a(x)
#     x = self.conv3_b(x)
#     x = self.pool3(x)
#     x = self.conv4_a(x)
#     x = self.conv4_b(x)
#     x = self.pool4(x)

#     # r3d
#     sc_r3d = x
#     x = self.r_conv1(x)
#     x = self.relu(self.r_conv1_bn(x))
#     x = self.r_conv2(x)
#     x = self.relu(self.r_conv2_bn(x))
#     x = self.r_conv3(x)
#     x = self.relu(self.r_conv3_bn(x))
#     x = self.r_conv4(x)
#     x = self.relu(self.r_conv4_bn(x) + sc_r3d)

#     # a3d
#     sc_a3d = x

#     shape = [x.shape[0],x.shape[1]*x.shape[2]*x.shape[3],int(self.reduction_channel)]

#     k = self.k_conv(x)
#     k = tf.reshape(k,shape)
#     q = self.q_conv(x)
#     q = tf.reshape(q,shape)

#     w_spatio_temporal = tf.einsum('b i c, b j c -> b i j', k, q)
#     w_channel = tf.einsum('b d i, b d j -> b i j', k, q)

#     w_spatio_temporal = self.softmax(w_spatio_temporal)

#     v = self.v_conv(x)
#     v = tf.reshape(v,shape)

#     # spatio_temporal
#     out = tf.einsum('b i f, b f j -> b i j', w_spatio_temporal, v)
#     out = tf.reshape(out,[x.shape[0], x.shape[1], x.shape[2], x.shape[3], int(self.reduction_channel)])
#     out = self.restore_conv(out)

#     x = out + sc_a3d

#     # classification(fc_out)
#     x = tf.reshape(x,[x.shape[0],-1])
#     x = self.fc6(x)
#     x = self.dropout(x)
#     x = self.fc7(x)
#     x = self.dropout(x)
#     x = self.fc8(x)

#     return x
