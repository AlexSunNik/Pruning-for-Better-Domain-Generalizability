PRUNED_NUMs = {}
# PRUNE_RATIO = 0.15
# PRUNE_RATIO = 0.05
# PRUNE_TIMES = 3
import re
MODULE_PAT = re.compile("(\d)")

import torch
import train
import mnist
import mnistm
import model
from utils import get_free_gpu
import argparse
import test
import numpy as np
from utils import set_model_mode
import matplotlib.pyplot as plt
from collections import defaultdict
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
ssim = SSIM(data_range=1.0)
from copy import deepcopy
import pandas as pd
import torch.nn as nn
from masking import *
from datetime import datetime
# *************************************************************************************** #
class DSS:
    def __init__(self):
        pass
        # mode 0: dot products of normalized
        # mode 1: simple mean l2
    def compute(self, feat_source, feat_target, mode=0):
        # B, C, H, W
        B, C, H, W = feat_source.shape
        if mode == 0:
            v1 = feat_source.view(B, C, -1) 
            v1 = v1 / torch.linalg.norm(v1, dim=-1).unsqueeze(-1)
            v2 = feat_target.view(B, C, -1) 
            v2 = v2 / torch.linalg.norm(v2, dim=-1).unsqueeze(-1)
            prod = (v1 * v2).sum(dim=-1)
            # Larger prod means more similar
            return prod
        elif mode == 1:
            return torch.mean(torch.sqrt((feat_source - feat_target) ** 2), [2, 3])
#         elif mode == 2:
#             return ssim(feat_source, feat_target)
# *************************************************************************************** #

def get_num_params(encoder):
    active = 0
    total = 0
    for k, v in encoder.named_modules():
        if hasattr(v, 'weight_mask'):
    #         print(v.weight_mask.shape)
            active += float(torch.sum(v.weight_mask != 0))
            total += v.weight_mask.numel()
        if hasattr(v, 'bias_mask'):
    #         print(v.bias_mask.shape)
            active += float(torch.sum(v.bias_mask != 0))
            total += v.bias_mask.numel()
    return active, total

# *************************************************************************************** #

parser = argparse.ArgumentParser(description="Baseline Training")
parser.add_argument('--prune', type=int, default=15)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--times', type=int, default=1)

args = parser.parse_args()
total_pruneratio = float(args.prune) / 100.0
density = 1 - total_pruneratio
epochs = args.epochs
PRUNE_TIMES = args.times
shrink = density ** (1/PRUNE_TIMES)
print(shrink)
PRUNE_RATIO = 1 - shrink
print(PRUNE_RATIO)
# *************************************************************************************** #

now = datetime.now()
date_time = now.strftime("%m-%d-%Y-H:%M:%S")
exp_name = f"l2_prunetimes{PRUNE_TIMES}_ratio{total_pruneratio}_epochs{epochs}_{date_time}"
log_file_name = f"logs/{exp_name}.txt"
log_file = open(log_file_name, 'w')

dss = DSS()

source_train_loader = mnist.mnist_train_loader
target_train_loader = mnistm.mnistm_train_loader
source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader

encoder = model.Extractor().cuda()
classifier = model.Classifier().cuda()
encoder_saved = deepcopy(encoder)

# Load Pretrained Model
encoder.load_state_dict(torch.load("./trained_models/encoder_source_mnist.pt"))
classifier.load_state_dict(torch.load("./trained_models/classifier_source_mnist.pt"))
# encoder.load_state_dict(torch.load("./trained_models/encoder_dann_omg.pt"))
# classifier.load_state_dict(torch.load("./trained_models/classifier_dann_omg.pt"))
def add_masking_hooks(model):
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            add_attr_masking(layer, 'weight', ParameterMaskingType.Soft, True)
            if hasattr(layer, "bias") and layer.bias is not None:
                add_attr_masking(layer, 'bias', ParameterMaskingType.Soft, True)

add_masking_hooks(encoder)
encoder.eval()
test.tester(encoder, classifier, None, source_test_loader, target_test_loader, 'prune', log_file=log_file)
encoder.train()
classifier.train()
# *************************************************************************************** #
for prune_iter in range(PRUNE_TIMES):
    print(f"Performing {prune_iter}th prune.")
    for name, module in encoder.named_modules():
        obj = MODULE_PAT.search(name)
        if hasattr(module, 'weight_orig'):
            name = int(obj.group(1))
            vals = module.weight_orig.data * module.weight_mask
            chn_num = vals.shape[0]
            scores = torch.sum(torch.abs(vals.view(chn_num, -1)), -1)
            val, idxs = torch.sort(scores)
            idxs = idxs.tolist()
            
            cur_prune_num = PRUNED_NUMs.get(name, 0)
            print("Current Pruned Number", cur_prune_num)
            PRUNE_NUM = int((chn_num - cur_prune_num) * PRUNE_RATIO)
            PRUNE_NUM += cur_prune_num
            PRUNED_NUMs[name] = PRUNE_NUM
        #         print(idxs)
            # print(idxs[:PRUNE_NUM])
            # print(chn_num)
            mask = torch.ones(chn_num)
            mask[idxs[:PRUNE_NUM]] = 0
            set_mask(encoder.extractor[name], 'weight', mask.view(-1, 1, 1, 1).cuda())
            set_mask(encoder.extractor[name], 'bias', mask.cuda())
    # *************************************************************************************** #
    print("Pre-finetuning")
    # Get number of parameters
    active, total = get_num_params(encoder)
    print(f"{active} parameters out of {total}")
    log_file.write(f"{active} parameters out of {total}\n\n")
    encoder.eval()
    test.tester(encoder, classifier, None, source_test_loader, target_test_loader, 'prune', log_file=log_file)
    encoder.train()
    # finetuning for a while
    train.finetune(encoder, classifier, source_train_loader, target_train_loader, save_name=f"{prune_iter}th_{exp_name}", epochs=epochs, log_file=log_file)

    print("Post-finetuning")
    encoder.eval()
    test.tester(encoder, classifier, None, source_test_loader, target_test_loader, 'prune', log_file=log_file)
    encoder.train()
    # *************************************************************************************** #

torch.save(encoder, f"{exp_name}_final_pruned_model.pt")
print(PRUNED_NUMs)