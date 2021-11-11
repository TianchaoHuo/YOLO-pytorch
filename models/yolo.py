

from .common import (Conv, Bottleneck, SPP, Concat, Detect, DWConv, 
                        MixConv2d, Focus, CrossConv, BottleneckCSP, C3 , NMS)
from .general import (scale_img, time_synchronized, check_anchor_order, 
                        initialize_weights,model_info, fuse_conv_and_bn)

import torch
import torch.nn as nn
from yaml import load
from pathlib import Path
from copy import deepcopy
import logging
import math

logger = logging.getLogger(__name__)



def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch): #model dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors # number of anchors
    no = na * (nc + 5) # number of outputs = anchors * (classes + 5)
    layers, save, c2 = [], [], ch[-1] # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']): # from, number, module, args
        
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):  # 把参数都读进来 args:[channel, kernel size, stride]
            try:
                args[j] = eval(a) if isinstance(a, str) else a 
            except:
                 pass
        
        n = max(round(n * gd), 1) if n > 1 else n # depth gain <--->depth_multiple

        
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # make_divisible(A,B)：找到比A大的，能整除B的最小整数。
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2 # 控制宽度gw

            args = [c1, c2, *args[1:]] # 將输入通道放入args列表
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args[ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x+1] for x in f])
            if isinstance(args[1], int): # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]


        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args) # module
        t = str(m) [8:-2]
        np = sum([x.numel() for x in m_.parameters()]) #number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1) # append to savelist
        layers.append(m_)
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class Model(nn.Module):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None): # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                #self.yaml = yaml.load(f, Loader=yaml.FullLoader) #model dict
                self.yaml = yaml.safe_load(f)  # model dict
        
        # Define model 
        if nc and nc != self.yaml['nc']: # 和数据集的类别数不一样
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc)) 
            self.yaml['nc'] = nc # override yaml value
        
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch]) # model, savelist, ch_out

        # Build strides, anchors
        m = self.model[-1] # Detect()
        #print(m.anchors)
        if isinstance(m, Detect):
            s = 416 # 2x min stride
            # tensor([ 8., 16., 32.])
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1) # 最大的anchors缩小32倍。
            # tensor([[[ 1.2500,  1.6250],
            #         [ 2.0000,  3.7500],
            #         [ 4.1250,  2.8750]],

            #         [[ 1.8750,  3.8125],
            #         [ 3.8750,  2.8125],
            #         [ 3.6875,  7.4375]],

            #         [[ 3.6250,  2.8125],
            #         [ 4.8750,  6.1875],
            #         [11.6562, 10.1875]]])
            check_anchor_order(m)
            self.stride = m.stride 
            self._initialize_biases() # only run once
        
        initialize_weights(self)
        self.info()
        print('')


    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:] # height, width
            s = [1, 0.83, 0.67] # scales
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0] # forward
                yi[..., :4] /= si # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1] # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0] # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None # augmented inference, train
        else: 
            # forward_once return a list, including three level feature maps
            # for input 416x416x3 
            # output shape [torch.Size([1, 3, 52, 52, 85]) 
            #               torch.Size([1, 3, 26, 26, 85]) 
            #               torch.Size([1, 3, 13, 13, 85])]
            return self.forward_once(x, profile) # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1: # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] # from earlier layers
            
            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 # FLOPS
                except:
                    o = 0
                
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        #print(len(x), x[0].shape, x[1].shape, x[2].shape)
        return x
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def add_nms(self):  # fuse model Conv2d() + BatchNorm2d() layers
        if type(self.model[-1]) is not NMS:  # if missing NMS
            print('Adding NMS module... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
        return self







if __name__ == "__main__":
    model = Model(cfg='./config/yolov3.yaml', ch=3, nc=80)
    