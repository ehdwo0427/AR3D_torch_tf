import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append('..')


# from config import Path

class r3d_module(nn.Module):
    def __init__(self):
        super(r3d_module, self).__init__()
        self.r_conv1 = nn.Conv3d(512, 128, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.r_conv1_bn = nn.BatchNorm3d(128)
        self.r_conv2 = nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.r_conv2_bn = nn.BatchNorm3d(128)
        self.r_conv3 = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.r_conv3_bn = nn.BatchNorm3d(128)
        self.r_conv4 = nn.Conv3d(128, 512, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.r_bn = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.r_conv1(x)
        x = self.r_conv1_bn(self.relu(x))
        x = self.r_conv2(x)
        x = self.r_conv2_bn(self.relu(x))
        x = self.r_conv3(x)
        x = self.r_conv3_bn(self.relu(x))
        x = self.r_conv4(x)
        out = self.r_bn(x)

        return out


class a3d_module(nn.Module):
    def __init__(self, reduction_ratio, attention_method):
        super(a3d_module, self).__init__()
        ## conv output dims? unknown reduction ratio
        self.reduction_channel = 512 // reduction_ratio
        self.attention_method = attention_method

        self.k_conv = nn.Conv3d(512, self.reduction_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.q_conv = nn.Conv3d(512, self.reduction_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.v_conv = nn.Conv3d(512, self.reduction_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.restore_conv = nn.Conv3d(self.reduction_channel, 512, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        if self.attention_method == 'channel':
            # attention weight extraction part
            # batch_size = x.shape[0]
            k = self.k_conv(x)
            q = self.q_conv(x)
            # reshape --> (batch_size, channel_number, featuresize(==2x7x7))
            k = k.view(x.shape[0], self.reduction_channel, k.shape[2] * k.shape[3] * k.shape[4])
            q = q.view(x.shape[0], self.reduction_channel, q.shape[2] * q.shape[3] * q.shape[4])
            w = torch.einsum('b i k, b j k -> b i j', k, q)
            w = self.softmax(w)

            # feature(value) extraction part
            v = self.v_conv(x)
            v = v.view(x.shape[0], self.reduction_channel, v.shape[2] * v.shape[3] * v.shape[4])

            out = torch.einsum('b i d, b d j -> b i j', w, v)
            out = out.view(x.shape[0], self.reduction_channel, x.shape[2], x.shape[3], x.shape[4])
            out = self.restore_conv(out)

        else:  ## self.attention_method =='spatio_temporal'
            k = self.k_conv(x)
            k = k.view(x.shape[0], self.reduction_channel, k.shape[2] * k.shape[3] * k.shape[4]).permute(0, 2, 1)
            q = self.q_conv(x)
            q = q.view(x.shape[0], self.reduction_channel, q.shape[2] * q.shape[3] * q.shape[4]).permute(0, 2, 1)
            v = self.v_conv(x)
            v = v.view(x.shape[0], self.reduction_channel, v.shape[2] * v.shape[3] * v.shape[4]).permute(0, 2, 1)
            w = torch.einsum('b i d, b j d -> b i j', k, q)
            w = self.softmax(w / np.sqrt(x.shape[2] * x.shape[3] * x.shape[4]))

            out = torch.einsum('b i d, b d j -> b i j', w, v).permute(0, 2, 1)
            out = out.view(x.shape[0], self.reduction_channel, x.shape[2], x.shape[3], x.shape[4])
            out = self.restore_conv(out)

        return out


class sfe_block(nn.Module):
    def __init__(self):
        super(sfe_block, self).__init__()
        ## SFE stage
        # SFE block 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # SFE block 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # SFE block 3
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # SFE block 4
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.relu = nn.ReLU()

    def forward(self, x):
        ## SFE 
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        out = self.pool4(x)

        return out


class c3d_conv(nn.Module):
    def __init__(self):
        super(c3d_conv, self).__init__()
        ## SFE stage + one extra convolution stage
        # SFE block 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # SFE block 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # SFE block 3
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # SFE block 4
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Extra convolution block (from original c3d model)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        ## SFE + one extra convolution stage
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        out = self.pool5(x)

        return out


class C3D(nn.Module):
    def __init__(self, num_classes):
        super(C3D, self).__init__()
        self.sfe = c3d_conv()
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.sfe(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AR3D(nn.Module):
    """
    The AR3D network.
    
    hyper params : reduction_ratio  -- default 4
                   hidden unit in prediction fc   -- default 4096
    SFE Type     : type 1  -->  from AR3D paper , 4 convolution stages  [default]
                   type 2  -->  from original c3d model, 5 convolution stages
    AR3D version : version 1 --> DFE module constructed with parallel Residual and Attention module
                   version 2 --> DFE module constructed with sequential Residual and Attention module in order [default]
    """

    def __init__(self, num_classes, AR3D_V='v2', SFE_type='t1', attention_method='spatio_temporal', reduction_ratio=4,
                 hidden_unit=4096):
        super(AR3D, self).__init__()
        ## sfe
        if SFE_type == 't1':
            ## SFE_type == 't1'
            ## sfe layer from ar3d paper (default)
            self.sfe = sfe_block()
            fc_input_size = 50176
        else:
            ## SFE_type == 't2'
            ## sfe layer from c3d's convolution layer
            self.sfe = c3d_conv()
            fc_input_size = 8192

        ## dfe
        self.version = AR3D_V
        self.residual = r3d_module()
        self.attention = a3d_module(reduction_ratio, attention_method)

        ## prediction (fc)
        self.fc6 = nn.Linear(fc_input_size, hidden_unit)
        self.fc7 = nn.Linear(hidden_unit, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
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

            sc = x
            x = self.residual(x)
            x = self.relu(x + sc)
            sc2 = x
            x = self.attention(x)
            x = x + sc2

            ## prediction (fc)
            ## flattening x --> (batch_size x 8192) for sfe_type2 
            ##                  (batch_size x 50176) for sfe_type1 
            x = x.view(x.shape[0], -1)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            x = self.dropout(x)
            logits = self.fc8(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params_c3d(model):
    b = [  ## SFE
        model.sfe.conv1, model.sfe.conv2, model.sfe.conv3a, model.sfe.conv3b, model.sfe.conv4a, model.sfe.conv4b,
        model.sfe.conv5a, model.sfe.conv5b,
        ## fc
        model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params_c3d(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def get_1x_lr_params(model, SFE_type='t1'):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    if SFE_type == 't1':
        b = [  ## SFE
            model.sfe.conv1, model.sfe.conv2, model.sfe.conv3a, model.sfe.conv3b, model.sfe.conv4a, model.sfe.conv4b,
            ## DFE1 - residual
            model.residual.r_conv1, model.residual.r_conv2, model.residual.r_conv3, model.residual.r_conv4,
            ## DFE2 - attention
            model.attention.k_conv, model.attention.q_conv, model.attention.v_conv, model.attention.restore_conv,
            ## fc
            model.fc6, model.fc7]
    else:
        b = [  ## SFE
            model.sfe.conv1, model.sfe.conv2, model.sfe.conv3a, model.sfe.conv3b, model.sfe.conv4a, model.sfe.conv4b,
            model.sfe.conv5a, model.sfe.conv5b,
            ## DFE1 - residual
            model.residual.r_conv1, model.residual.r_conv2, model.residual.r_conv3, model.residual.r_conv4,
            ## DFE2 - attention
            model.attention.k_conv, model.attention.q_conv, model.attention.v_conv, model.attention.restore_conv,
            ## fc
            model.fc6, model.fc7]

    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    # net = AR3D(num_classes=101, pretrained=False)
    net = AR3D(num_classes=101, AR3D_V='v2', SFE_type='t1', attention_method='spatio_temporal', reduction_ratio=4,
               hidden_unit=4096)

    outputs = net.forward(inputs)
    print(outputs.size())
