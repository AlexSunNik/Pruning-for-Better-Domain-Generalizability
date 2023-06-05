PRUNED_NUMs = {}
# PRUNE_RATIO = 0.15
# PRUNE_RATIO = 0.05
# PRUNE_TIMES = 3

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
exp_name = f"method2perdss_prunetimes{PRUNE_TIMES}_ratio{total_pruneratio}_epochs{epochs}_{date_time}"
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
    scores = defaultdict(list)
    scores1 = defaultdict(list)
    # Get some data
    # *************************************************************************************** #
    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Source input -> Source Classification
        source_image, source_label = source_data
        source_image, source_label = source_image, source_label
        source_image, source_label = source_image.cuda(), source_label.cuda()
        source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
        source_feature, source_fmaps = encoder(source_image, return_feat=True)
        source_output = classifier(source_feature)
        source_pred = source_output.data.max(1, keepdim=True)[1]


        # 2. Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image, target_label
        target_image, target_label = target_image.cuda(), target_label.cuda()
        target_feature, target_fmaps = encoder(target_image, return_feat=True)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]

        
        # Compute scores
        for name, fmap in source_fmaps.items():
            target_fmap = target_fmaps[name]
            chn_num = fmap.shape[1]
            score = dss.compute(fmap, target_fmap, mode=0) #B, C
            score1 = dss.compute(fmap, target_fmap, mode=1) #B, C
            scores[name].append(score)
            scores1[name].append(score1)
    # *************************************************************************************** #
    scores = {k:torch.cat(v, 0) for k, v in scores.items()}
    scores1 = {k:torch.cat(v, 0) for k, v in scores1.items()}
    scores = {k:torch.nan_to_num(v, 0) for k, v in scores.items()}
    scores1 = {k:torch.nan_to_num(v, 0) for k, v in scores1.items()}
    avg_score = {k:v.mean(0) for k, v in scores.items()}
    avg_score1 = {k:v.mean(0) for k, v in scores1.items()}
    # *************************************************************************************** #
    # Perform Pruning
    # Use masking
    for name, fmap in source_fmaps.items():
        # print(name)
        val, idxs = torch.sort(torch.nan_to_num(torch.tensor(avg_score[name]), nan=-10))
        idxs = idxs.tolist()
    #     idxs = idxs[::-1]
        # print(len(idxs))
        target_fmap = target_fmaps[name]
        print("For layer", name)
        chn_num = fmap.shape[1]
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
        set_mask(encoder.extractor[name-1], 'weight', mask.view(-1, 1, 1, 1).cuda())
        set_mask(encoder.extractor[name-1], 'bias', mask.cuda())
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