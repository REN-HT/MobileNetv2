import torch
from config import opt
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, t=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.one_conv=nn.Sequential(nn.Conv2d(inchannel, inchannel*t, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(inchannel*t),
                                    nn.ReLU6())

        self.two_conv=nn.Sequential(nn.Conv2d(inchannel*t, inchannel*t, 3, stride, 1, bias=False, groups=inchannel*t),
                                    nn.BatchNorm2d(inchannel*t),
                                    nn.ReLU6())

        self.three_conv=nn.Sequential(nn.Conv2d(inchannel*t, outchannel, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.one_conv(x)
        out = self.two_conv(out)
        out = self.three_conv(out)
        if self.stride == 1 and self.inchannel == self.outchannel:
            out += x
        return out


class make_layer(nn.Module):
    def __init__(self, inchannel, outchannel, t, nums_block, stride):
        super(make_layer, self).__init__()
        layers = []
        layers.append(Bottleneck(inchannel, outchannel, stride, t))
        for i in range(1, nums_block):
            layers.append(Bottleneck(outchannel, outchannel, 1, t))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class mobileNetv2(nn.Module):
    def __init__(self):
        super(mobileNetv2, self).__init__()
        self.num_classes=opt.nums_class
        self.conv2d=nn.Conv2d(3,32,3,2,1,bias=False)
        self.layer1=make_layer(32,16,1,1,1)
        self.layer2=make_layer(16,24,6,2,2)
        self.layer3=make_layer(24,32,6,3,2)
        self.layer4=make_layer(32,64,6,4,2)
        self.layer5=make_layer(64,96,6,3,1)
        self.layer6=make_layer(96,160,6,3,2)
        self.layer7=make_layer(160,320,6,1,1)
        self.conv=nn.Conv2d(320,1280,1,1,0)
        self.fcn=nn.Conv2d(1280,self.num_classes,1,1,0)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.conv(out)

        out = F.avg_pool2d(out, 7)
        out = F.dropout2d(out, 0.1)
        out = self.fcn(out)
        out = out.view(out.size(0), -1)
        # out = F.softmax(out, dim=1)
        return out


# from torch.autograd import Variable as V
# net=mobileNetv2()
# input=V(torch.randn(1,3,224,224))
# output=net(input)
# print(output)
#
# total=0
# for name, parameter in net.named_parameters():
#     temp=parameter.size()
#     qq=1
#     for i in range(len(temp)):
#         qq*=temp[i]
#     total+=qq
# print(total)

