import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF
from copy import deepcopy

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

# ************************************************************************************************ #
# Big Ones
class Extractor2(nn.Module):
    def __init__(self):
        super(Extractor2, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2), #1
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2), #2
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, padding=2), #2
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                if isinstance(module, nn.ReLU):
                    feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

class Extractor3(nn.Module):
    def __init__(self):
        super(Extractor3, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

        
class Extractor4(nn.Module):
    def __init__(self):
        super(Extractor4, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

class Extractor5(nn.Module):
    def __init__(self):
        super(Extractor5, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

class Extractor6(nn.Module):
    def __init__(self):
        super(Extractor6, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats
        
# ************************************************************************************************ #
# Small Ones
class Extractor7(nn.Module):
    def __init__(self):
        super(Extractor7, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

class Extractor8(nn.Module):
    def __init__(self):
        super(Extractor8, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

class Extractor9(nn.Module):
    def __init__(self):
        super(Extractor9, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=8, out_channels=48, kernel_size=5, padding=2), #1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x, return_feat=False, keep_grad=False):
        if not return_feat:
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
            return x
        else:
            feats = {}
            for name, module in enumerate(self.extractor):
                x = module(x)
                # print(x.shape)
                if isinstance(module, nn.ReLU):
                    if keep_grad:
                        feats[name] = x
                    else: 
                        feats[name] = deepcopy(x.data)
            x = x.view(-1, 3 * 28 * 28)
            return x, feats

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x)
