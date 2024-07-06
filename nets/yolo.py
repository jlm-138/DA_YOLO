import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class SpatialAttention(nn.Module):  # 源域敏感
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttention2(nn.Module):    # 目标域敏感
    def __init__(self, kernel_size=3):
        super(SpatialAttention2, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MyChannel(nn.Module):
    def __init__(self, num_channels):
        super(MyChannel, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_channels, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.linear(x.view(x.size(0), -1))
        x = x.view(x.size(0), 2, 1, 1)
        x = self.softmax(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, with_sigmoid=False):
        super(SELayer, self).__init__()
        self.with_sigmoid = with_sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if with_sigmoid:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class DomainAttention(nn.Module):
    def __init__(self, planes, reduction=16):
        super(DomainAttention, self).__init__()
        self.planes = planes
        num_adapters = 2  # 源域和目标域两个

        self.n_datasets = num_adapters
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.SE_Layers = nn.ModuleList([SELayer(planes, reduction) for num_class in range(num_adapters)])
        self.fc_1 = nn.Linear(planes, self.n_datasets)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()

        weight = self.fc_1(self.avg_pool(x).view(b, c))
        weight = self.softmax(weight).view(b, self.n_datasets, 1)
        for i, SE_Layer in enumerate(self.SE_Layers):
            if i == 0:
                SELayers_Matrix = SE_Layer(x).view(b, c, 1)
            else:
                SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)  # C*N
        SELayers_Matrix = torch.matmul(SELayers_Matrix, weight).view(b, c, 1, 1)  # C*N 乘 N*1
        SELayers_Matrix = self.sigmoid(SELayers_Matrix)

        return x*SELayers_Matrix


class MyDa(nn.Module):
    def __init__(self, input_c, c1=512, c2=256, c3=128, c4=64):
        super(MyDa, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_c, out_channels=c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)  # 添加BN层

        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)  # 添加BN层

        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)  # 添加BN层

        self.conv4 = nn.Conv2d(in_channels=c3, out_channels=c4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c4)  # 添加BN层
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.15, inplace=True)

        self.conv5 = nn.Conv2d(in_channels=c4, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # 使用BN层
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)  # 使用BN层
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)  # 使用BN层
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x)  # 使用BN层
        x = self.leaky_relu(x)

        x = self.conv5(x)

        return x  # 归一化到0-1


class MyDa_one(nn.Module):
    def __init__(self,input_c,c1=256,c2=128,c3=64):
        super(MyDa_one, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_c,out_channels=c1, kernel_size=3, padding=1)  #
        self.bn1 = nn.BatchNorm2d(c1)  # 添加BN层

        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)  # 添加BN层

        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)  # 添加BN层
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.15, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=c3, out_channels=1, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(1)  # 添加BN层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化层

    def forward(self, x):
        # batch_size, c,w, h = x.size()
        x = self.conv1(x)
        x=self.bn1(x)
        x=self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x=self.leaky_relu(x)

        x=self.conv3(x)
        x = self.bn3(x)
        x=self.leaky_relu(x)

        x=self.conv4(x)
        x=self.global_avg_pool(x)

        # return x  # 归一化到0-1
        return torch.sigmoid(x)   # 归一化到0-1


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy         = deploy
        self.groups         = g
        self.in_channels    = c1
        self.out_channels   = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11  = autopad(k, p) - k // 2
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam    = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity   = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense      = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1        = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3  = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1  = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid    = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel      = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma       = branch[1].weight
            beta        = branch[1].bias
            eps         = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel      = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma       = branch.weight
            beta        = branch.bias
            eps         = branch.eps
        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std     = (bn.running_var + bn.eps).sqrt()
        bias    = bn.bias - bn.running_mean * bn.weight / std

        t       = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn      = nn.Identity()
        conv    = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias   = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense  = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1    = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias    = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1           = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded      = identity_conv_1x1.bias
            weight_identity_expanded    = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded      = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded    = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        
        self.rbr_dense.weight   = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias     = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam    = self.rbr_dense
        self.deploy         = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None
            
def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov7版本的参数
        #-----------------------------------------------#
        transition_channels = {'l' : 32, 'x' : 40}[phi]
        block_channels      = 32
        panet_channels      = {'l' : 32, 'x' : 64}[phi]
        e       = {'l' : 2, 'x' : 1}[phi]
        n       = {'l' : 4, 'x' : 6}[phi]
        ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]
        conv    = {'l' : RepConv, 'x' : Conv}[phi]
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #---------------------------------------------------#
        self.backbone   = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        #------------------------加强特征提取网络------------------------# 
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 20, 20, 1024 => 20, 20, 512
        self.sppcspc                = SPPCSPC(transition_channels * 32, transition_channels * 16)
        # 20, 20, 512 => 20, 20, 256 => 40, 40, 256
        self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        # 40, 40, 1024 => 40, 40, 256
        self.conv_for_feat2         = Conv(transition_channels * 32, transition_channels * 8)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        # 40, 40, 256 => 40, 40, 128 => 80, 80, 128
        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        # 80, 80, 512 => 80, 80, 128
        self.conv_for_feat1         = Conv(transition_channels * 16, transition_channels * 4)
        # 80, 80, 256 => 80, 80, 128
        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        # 80, 80, 128 => 40, 40, 256
        self.down_sample1           = Transition_Block(transition_channels * 4, transition_channels * 4)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        # 40, 40, 256 => 20, 20, 512
        self.down_sample2           = Transition_Block(transition_channels * 8, transition_channels * 8)
        # 20, 20, 1024 => 20, 20, 512
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)
        #------------------------加强特征提取网络------------------------# 

        # 80, 80, 128 => 80, 80, 256
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        # 40, 40, 256 => 40, 40, 512
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        # 20, 20, 512 => 20, 20, 1024
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        # 4 + 1 + num_classes
        # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * 25 & 85
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 512 => 20, 20, 3 * 25 & 85
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)


        self.myda1=MyDa(transition_channels * 16)   # 512
        self.myda2 = MyDa(transition_channels * 32)  # 1024
        self.myda3 = MyDa(transition_channels * 32)

        self.myda1_one=MyDa_one(transition_channels * 8)  # 256
        self.myda2_one=MyDa_one(transition_channels * 16) # 512
        self.myda3_one=MyDa_one(transition_channels * 32)

        self.myw=SpatialAttention()
        self.myw2 = SpatialAttention2()

        self.myc1=MyChannel(transition_channels * 16)  # 512
        self.myc2 = MyChannel(transition_channels * 32)
        self.myc3 = MyChannel(transition_channels * 32)


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        #------------------------加强特征提取网络------------------------# 
        # 20, 20, 1024 => 20, 20, 512
        P5          = self.sppcspc(feat3)   # P5
        # 20, 20, 512 => 20, 20, 256
        P5_conv     = self.conv_for_P5(P5)
        # 20, 20, 256 => 40, 40, 256
        P5_upsample = self.upsample(P5_conv)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        # 40, 40, 512 => 40, 40, 256
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 256 => 40, 40, 128
        P4_conv     = self.conv_for_P4(P4)
        # 40, 40, 128 => 80, 80, 128
        P4_upsample = self.upsample(P4_conv)
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        # 80, 80, 256 => 80, 80, 128
        P3          = self.conv3_for_upsample2(P3)

        # 80, 80, 128 => 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 => 40, 40, 256
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 256 => 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 => 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 => 20, 20, 512
        P5 = self.conv3_for_downsample2(P5)
        #------------------------加强特征提取网络------------------------# 
        # P3 80, 80, 128 
        # P4 40, 40, 256
        # P5 20, 20, 512
        
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # print('p3_shape',P3.shape)   # [2, 256, 80, 80])

        # print('da2_shape',da2.shape)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # print('p4_shape', P4.shape)
        # print('da1_shape',da1.shape)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        #   f1 80, 80, 512
        #   f2 40, 40, 1024
        #   f3 20, 20, 1024

        # print("feat1",feat1.shape)  # b,c,w,h
        # [空间注意力；空间注意力]*w_channel
        da_w2=torch.sum(torch.cat((self.myw(feat1),self.myw2(feat1)),dim=1)*self.myc1(feat1),dim=1,keepdim=True)
        da_w1=torch.sum(torch.cat((self.myw(feat2),self.myw2(feat2)),dim=1)*self.myc2(feat2),dim=1,keepdim=True)
        da_w0=torch.sum(torch.cat((self.myw(feat3),self.myw2(feat3)),dim=1)*self.myc3(feat3),dim=1,keepdim=True)
        # print(da_w2.shape)

        da2 = self.myda1(grad_reverse(feat1))  # 调整为batch,c,w,h
        da1 = self.myda2(grad_reverse(feat2))
        da0 = self.myda3(grad_reverse(feat3))

        # P3 80, 80, 256
        # P4 40, 40, 512
        # P5 20, 20, 1024
        da2_one = self.myda1_one(grad_reverse(P3))  # 调整为batch,c,w,h
        da1_one = self.myda2_one(grad_reverse(P4))
        da0_one = self.myda3_one(grad_reverse(P5))
        # print('000',da0_one.shape)
        # print('111',da1_one.shape)

        return [out0, out1, out2],[da0,da1,da2],[da0_one,da1_one,da2_one],[da_w0,da_w1,da_w2]
