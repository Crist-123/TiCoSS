import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

### network ###
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output


class upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch, reshape=True,scale_factor=2):
        super(upsample_layer, self).__init__()
        self.reshape = reshape
        self.scale_factor=scale_factor
        if reshape:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.reshape:
            x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output

class MyConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(MyConvLayer, self).__init__()
        self.myconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.myconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class GFF(nn.Module):
    def __init__(self,input_dim,ist_true=False):
        super(GFF, self).__init__()
        self.input_dim=input_dim
        if ist_true==False:
            self.remap = add_conv(self.input_dim//2, self.input_dim, 3, 2)
        else:
            self.remap=add_conv(self.input_dim//4,input_dim,3,2)
        self.upper_level_gate = add_conv(self.input_dim, self.input_dim, 1,1)
        self.now_level_gate = add_conv(self.input_dim, self.input_dim, 1,1)
        self.now_level_gate_2 = add_conv(self.input_dim, self.input_dim, 1,1)

    def forward(self, x_level_0, x_level_1):
        x0_n=self.remap(x_level_0)
        g0=self.upper_level_gate(x0_n)
        g0=torch.sigmoid(g0)
        x1_n=self.now_level_gate(x_level_1)
        g1=self.now_level_gate_2(x1_n)
        g1=torch.sigmoid(g1)
        xgff = (1+g1)*x1_n + (1-g1)*(g0*x0_n) #1,256,32,64 * 1,256,32,64
        return xgff
class RoadSeg_decoder(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self, num_labels, use_sne):
        super(RoadSeg_decoder, self).__init__()

        self.num_resnet_layers = 152
        self.gff_01=GFF(256,True)
        self.gff_12=GFF(512)
        self.gff_23=GFF(1024)
        self.gff_34=GFF(2048)
        self.seggff_01=GFF(256,True)
        self.seggff_12=GFF(512)
        self.seggff_23=GFF(1024)
        self.seggff_34=GFF(2048)
        self.myconv1 = MyConvLayer(64, 64, 3, 2, 1)
        self.myconv2 = MyConvLayer(96, 256, 3, 2, 1)
        self.myconv3 = MyConvLayer(256, 512, 3, 2, 1)
        
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = torchvision.models.resnet18(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet18(pretrained=True)
            filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet34(pretrained=True)
            filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = torchvision.models.resnet50(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet50(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = torchvision.models.resnet101(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet101(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = torchvision.models.resnet152(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet152(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError('num_resnet_layers should be 18, 34, 50, 101 or 152')

        self.guide1=nn.Sequential(
            conv_block_nested(filters[0],filters[1],filters[1]),
            nn.MaxPool2d(2,2)
        )
        self.guide2=nn.Sequential(
            conv_block_nested(filters[1],filters[2],filters[2]),
            nn.MaxPool2d(2,2)
        )
        self.guide3=nn.Sequential(
            conv_block_nested(filters[2],filters[3],filters[3]),
            nn.MaxPool2d(2,2)
        )
        self.guide4=nn.Sequential(
            conv_block_nested(filters[3],filters[4],filters[4]),
            nn.MaxPool2d(2,2)
        )
        # self.rguide1=nn.Sequential(
        #     conv_block_nested(filters[0],filters[1],filters[1]),
        #     nn.MaxPool2d(2,2)
        # )
        # self.rguide2=nn.Sequential(
        #     conv_block_nested(filters[1],filters[2],filters[2]),
        #     nn.MaxPool2d(2,2)
        # )
        # self.rguide3=nn.Sequential(
        #     conv_block_nested(filters[2],filters[3],filters[3]),
        #     nn.MaxPool2d(2,2)
        # )
        # self.rguide4=nn.Sequential(
        #     conv_block_nested(filters[3],filters[4],filters[4]),
        #     nn.MaxPool2d(2,2)
        # )
        # self.conv5=conv_block_nested(filters[4],filters[4],filters[4])
        ### encoder for another image ###
        if use_sne:
            self.encoder_another_conv1 = resnet_raw_model1.conv1
        else:
            # if another image is depth, initialize the weights of the first layer
            self.encoder_another_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder_another_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)

        self.encoder_another_bn1 = resnet_raw_model1.bn1
        self.encoder_another_relu = resnet_raw_model1.relu
        self.encoder_another_maxpool = resnet_raw_model1.maxpool
        self.encoder_another_layer1 = resnet_raw_model1.layer1
        self.encoder_another_layer2 = resnet_raw_model1.layer2
        self.encoder_another_layer3 = resnet_raw_model1.layer3
        self.encoder_another_layer4 = resnet_raw_model1.layer4

        ###  encoder for rgb image  ###
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ###  decoder  ###
        
        self.conv1_1 = conv_block_nested(filters[0]*2, filters[0], filters[0])
        self.conv2_1 = conv_block_nested(filters[1]*2, filters[1], filters[1])
        self.conv3_1 = conv_block_nested(filters[2]*2, filters[2], filters[2])
        self.conv4_1 = conv_block_nested(filters[3]*3, filters[3], filters[3])

        self.conv1_2 = conv_block_nested(filters[0]*3, filters[0], filters[0])
        self.conv2_2 = conv_block_nested(filters[1]*3, filters[1], filters[1])
        self.conv3_2 = conv_block_nested(filters[2]*4, filters[2], filters[2])

        self.conv1_3 = conv_block_nested(filters[0]*4, filters[0], filters[0])
        self.conv2_3 = conv_block_nested(filters[1]*5, filters[1], filters[1])

        self.conv1_4 = conv_block_nested(filters[0]*6, filters[0], filters[0])

        self.up2_0 = upsample_layer(filters[1], filters[0])
        self.up2_1 = upsample_layer(filters[1], filters[0])
        self.up2_2 = upsample_layer(filters[1], filters[0])
        self.up2_3 = upsample_layer(filters[1], filters[0])

        self.up3_0 = upsample_layer(filters[2], filters[1])
        self.up3_1 = upsample_layer(filters[2], filters[1])
        self.up3_2 = upsample_layer(filters[2], filters[1])

        self.up4_0 = upsample_layer(filters[3], filters[2])
        self.up4_1 = upsample_layer(filters[3], filters[2])

        self.up5_0 = upsample_layer(filters[4], filters[3])

        self.final = upsample_layer(filters[0], num_labels)
        self.final_1 = upsample_layer(filters[0], num_labels)
        self.final_2 = upsample_layer(filters[0], num_labels)
        self.final_3 = upsample_layer(filters[0], num_labels)
        self.finala=nn.Sequential(
            upsample_layer(filters[1], num_labels,scale_factor=4),
        )
        self.finalb=nn.Sequential(
            upsample_layer(filters[2], num_labels,scale_factor=8),
        )
        self.finalc=nn.Sequential(
            upsample_layer(filters[3], num_labels,scale_factor=16)
        )
        ### layers without pretrained model need to be initialized ###
        self.need_initialization = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv1_2,
                                    self.conv2_2, self.conv3_2, self.conv1_3, self.conv2_3, self.conv1_4,
                                    self.up2_0, self.up2_1, self.up2_2, self.up2_3, self.up3_0, self.up3_1,
                                    self.up3_2, self.up4_0, self.up4_1, self.up5_0, self.final]
    def forward(self, x1, x2, x3, another, test_mode=False):
        # encoder
        another = self.encoder_another_conv1(another)
        another = self.encoder_another_bn1(another)
        another_0 = self.encoder_another_relu(another)

        another_1 = self.encoder_another_maxpool(another_0)
        another_1 = self.encoder_another_layer1(another_1)
        another_1=self.gff_01(another_0,another_1)

        another_2= self.encoder_another_layer2(another_1)
        another_2=self.gff_12(another_1,another_2)

        another_3= self.encoder_another_layer3(another_2)
        another_3=self.gff_23(another_2,another_3)
        another_4= self.encoder_another_layer4(another_3)
        another_4=self.gff_34(another_3,another_4)

        rgb_0 = self.myconv1(x1)
        x1_0=rgb_0+another_0

        rgb_1 = self.myconv2(x2)
        x2_0=rgb_1+another_1
        x2_0=self.seggff_01(x1_0,x2_0)
        # x2_0=rgb_1+another_1

        rgb_2 = self.myconv3(x3)
        x3_0=rgb_2+another_2
        x3_0=self.seggff_12(x2_0,x3_0)
        # x3_0=rgb_2+another_2

        rgb_3 = self.encoder_rgb_layer3(x3_0)
        x4_0=rgb_3+another_3
        x4_0=self.seggff_23(x3_0,x4_0)
        # x4_0=rgb_3+another_3

        rgb_4=self.encoder_rgb_layer4(x4_0)
        x5_0=rgb_4+another_4
        x5_0=self.seggff_34(x4_0,x5_0)

        # x5_0=rgb_4+another_4
        #guide
        
        guide2=self.guide1(x1_0)
        guide3=self.guide2(guide2)
        guide4=self.guide3(guide3)
        guide5=self.guide4(guide4)

        


        # decoder
        # x5_0=x5_0+guide5

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up5_0(x5_0),guide4], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up4_1(x4_1),guide3], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up3_2(x3_2),guide2], dim=1))

        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up2_3(x2_3),x1_0], dim=1))
        out = self.final(x1_4)
        if test_mode:
            return out
        else:
            out11=self.final_1(x1_1)
            out12=self.final_2(x1_2)
            out13=self.final_3(x1_3)
            out1=self.finala(x2_3)
            out2=self.finalb(x3_2)
            out3=self.finalc(x4_1)
            return out,out1,out2,out3,out11,out12,out13