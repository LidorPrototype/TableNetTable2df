import torch
import torch.nn as nn
from Training.encoder import VGG19, DenseNet, ResNet, efficientNet


class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = kernels[0], stride = strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(in_channels = 256, out_channels=128, kernel_size = kernels[1], stride = strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(in_channels = 128 + channels[0], out_channels = 256, kernel_size = kernels[2], stride = strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(in_channels = 256 + channels[1], out_channels = 1, kernel_size = kernels[3], stride = strides[3])

    def forward(self, x, pool3_out, pool4_out):
        x = self.conv_7_table(x)
        out = self.upsample_1_table(x)
        out = torch.cat((out, pool4_out), dim=1)
        out = self.upsample_2_table(out)
        out = torch.cat((out, pool3_out), dim=1)
        out = self.upsample_3_table(out)
        return out

class ColumnDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(ColumnDecoder, self).__init__()
        self.conv_8_column = nn.Sequential(
            nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = kernels[0], stride = strides[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = kernels[0], stride = strides[0])
        )
        self.upsample_1_column = nn.ConvTranspose2d(in_channels = 256, out_channels=128, kernel_size = kernels[1], stride = strides[1])
        self.upsample_2_column = nn.ConvTranspose2d(in_channels = 128 + channels[0], out_channels = 256, kernel_size = kernels[2], stride = strides[2])
        self.upsample_3_column = nn.ConvTranspose2d( in_channels = 256 + channels[1], out_channels = 1, kernel_size = kernels[3], stride = strides[3])

    def forward(self, x, pool3_out, pool4_out):
        x = self.conv_8_column(x)
        out = self.upsample_1_column(x)
        out = torch.cat((out, pool4_out), dim=1)
        out = self.upsample_2_column(out)
        out = torch.cat((out, pool3_out), dim=1)
        out = self.upsample_3_column(out)
        return out

class TableNet(nn.Module):
    def __init__(self,encoder = 'vgg', use_pretrained_model = True, basemodel_requires_grad = True):
        super(TableNet, self).__init__()
        self.kernels = [(1,1), (2,2), (2,2),(8,8)]
        self.strides = [(1,1), (2,2), (2,2),(8,8)]
        self.in_channels = 512
        if encoder == 'vgg':
            self.base_model = VGG19(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [512, 256]
        elif encoder == 'resnet':
            self.base_model = ResNet(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [256, 128]
        elif encoder == 'densenet':
            self.base_model = DenseNet(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [512, 256]
            self.in_channels = 1024
            self.kernels = [(1,1), (1,1), (2,2),(16,16)]
            self.strides = [(1,1), (1,1), (2,2),(16,16)]
        elif 'efficientnet' in encoder:
            self.base_model = efficientNet(model_type = encoder, pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            if 'b0' in encoder:
                self.pool_channels = [192, 192]
                self.in_channels = 320
            elif 'b1' in encoder:
                self.pool_channels = [320, 192]
                self.in_channels = 320
            elif 'b2' in encoder:
                self.pool_channels = [352, 208]
                self.in_channels = 352
            self.kernels = [(1,1), (1,1), (1,1),(32,32)]
            self.strides = [(1,1), (1,1), (1,1),(32,32)]
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8)
        )
        self.table_decoder = TableDecoder(self.pool_channels, self.kernels, self.strides)
        self.column_decoder = ColumnDecoder(self.pool_channels, self.kernels, self.strides)

    def forward(self, x):
        pool3_out, pool4_out, pool5_out = self.base_model(x)
        conv_out = self.conv6(pool5_out)
        table_out = self.table_decoder(conv_out, pool3_out, pool4_out)
        column_out = self.column_decoder(conv_out, pool3_out, pool4_out)
        return table_out, column_out
