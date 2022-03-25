from efficientnet_pytorch import EfficientNet
import timm
import torchvision.models as models
import torch.nn as nn
import pretrainedmodels
def model_set(args):

    if args.models == 'resnet18':

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)
    elif args.models == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)
    elif args.models == 'resnet101':
        model_ft = models.resnet101(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)
    elif args.models =='mobilenet':
        model_ft = models.mobilenet_v2(pretrained=True)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, args.nc)
    elif args.models =='efficient':
        model_name = 'efficientnet-b1'  # b0 ~ b7
        image_size = EfficientNet.get_image_size(model_name)
        model_ft = EfficientNet.from_pretrained(model_name, num_classes=args.nc)
    elif args.models =='wide50':
        model_ft = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)
    elif args.models =='wide101':
        model_ft = models.wide_resnet101_2(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)
    elif args.models =='resnext101':
        model_ft = models.resnext101_32x8d(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)    
    elif args.models =='resnext50':
        model_ft = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.nc)   
    elif args.models == 'swin-tiny':
        model_ft = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=args.nc)
    elif args.models == 'swin-base':
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=args.nc)
    elif args.models == 'swin-small':
        model_ft = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=args.nc)
    elif args.models == 'swin-large':
        model_ft = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=args.nc)
    elif args.models =='s3d-tiny' or args.models =='s3d-large':
        model_ft = S3D(args)

    return model_ft
    
    
class S3D(nn.Module):

    def __init__(self, args):
        super(S3D, self).__init__()
        

        """3D Block"""

        self.conv3D_0 = nn.Conv3d(3, 16, kernel_size=3, padding=(1,1,1), stride=(4,1,1))
        self.conv3D_1 = nn.Conv3d(16,32, kernel_size=3, padding=(1,1,1), stride=(2,1,1))
        self.conv3D_2 = nn.Conv3d(32,3, kernel_size=3, padding=(1,1,1), stride=(2,1,1))



        """Batch Norm"""
        self.batch16 = nn.BatchNorm3d(16)
        self.batch32 = nn.BatchNorm3d(32)
        self.batch64 = nn.BatchNorm3d(64)

        self.batch3 = nn.BatchNorm3d(3)

        """ Activation Function """

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        #self.model_2d = torch.load('./models/densenet_5frame_spatial.pth', map_location='cuda:0')

        if args.models == 's3d-tiny':
            self.model_ft = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=args.nc)
        elif args.models == 's3d-large':
            self.model_ft = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=args.nc)
        
        self.model_3d = nn.Sequential(self.conv3D_0, self.batch16, self.relu, self.conv3D_1, self.batch32, self.relu, self.conv3D_2, self.batch3, self.relu)

    def forward(self, x):

        X = self.model_3d(x)
        #print(x.shape)
        #x = self.relu(self.batch16(self.conv3D_0(x)))
        #x = self.relu(self.batch32(self.conv3D_1(x)))
        #x = self.relu(self.batch3(self.conv3D_2(x)))


        #print(x.shape, "F")
        """ 
        3D CNN's output shape is (16,1,3,224,224)

        So change the shape to (16,3,224,224) to match 2D CNN's input
        """

        x = x.reshape(-1, 3, x.shape[3], x.shape[4])

        x_1, _,_,_ = x.shape
        # np save 후 다시 load했을 때 성능 동일한지만 확인
        #for k in range(int(x_1)):
        #    x_each = x[k]

        #    np.save("%d.npy"%k,x_each)

        x = self.model_ft(x) 

        return x
    
    
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        #last_duration = 1
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
    
    
