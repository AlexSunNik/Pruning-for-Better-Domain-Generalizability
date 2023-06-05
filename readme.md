# Pruning for Better Domain Generalizability

This repository is for method introduced in the following paper:

Pruning for Better Domain Generalizability\
Xinglong Sun

## Introduction
n this paper, we investigate whether we could
use pruning as a reliable method to boost the
generalization ability of the model. We found
that existing pruning method like L2 can already
offer small improvement on the target domain
performance. We further propose a novel prun-
ing scoring method, called DSS, designed not
to maintain source accuracy as typical pruning
work, but to directly enhance the robustness of
the model. We conduct empirical experiments to
validate our method and demonstrate that it can
be even combined with state-of-the-art generaliza-
tion work like MIRO(Cha et al., 2022) to further
boost the performance. On MNIST to MNIST-M,
we could improve the baseline performance by
over 5 points by introducing 60% channel spar-
sity into the model. On DomainBed benchmark
and state-of-the-art MIRO, we can further boost
its performance by 1 point only by introducing
10% sparsity into the model.

<!-- <div align="center">
  <img src="Figs/flowchart_resize.png" width="100%">
  Overview of our method.
</div> -->

## Prerequisites
### Datasets

For MNIST-M:\
Should be autodownloaded when running the code. Let me know if it does not.

For DomainBed, please run:
```
cd domainbed_miro_exp
python3 -m domainbed.scripts.download --data_dir=[DATASET_PATH]
```

I also provide pretrained models and DSS scores in:\
https://drive.google.com/file/d/17MbHJo-khjhxo04Yh6XRPiA-EDRdYfbU/view?usp=sharing
After unzipping the folder:
1. move the two '.pt' files inside mnist_m_exp into mnist_m_exp/trained_models.
2. move domainbed_miro_exp/PACS/PACS_*_score.pkl into domainbed_miro_exp/.
3. move domainbed_miro_exp/PACS/*.pth into domainbed_miro_exp/train_output/PACS.
4. repeat Step 2 but for OfficeHome
5. repeat Step 3 but for OfficeHome

## MNIST-M
All experiments and related code can be found in mnist_m_exp

Run first:
```
cd mnist_m_exp
```

For Baseline:
```
python3 main.py
```
I also provide the pretrained baseline by us in trained_models

Results in attached Figure 2

With the DSS:
```
bash run_diff_pruneratio_dss.sh
```
With L2:
```
bash run_diff_pruneratio_dss.sh
```
For reproduction purpose, I also provide training logs in mnist_m_exp/logs

## DomainBed
All experiments and related code can be found in domainbed_miro_exp

Run first:
```
cd mnist_m_exp
```

For baseline MIRO
```
bash baselines.sh
```
I also include the pretrained baselines inside train_output/PACS/\*_final.pth and train_output/OfficeHome/\*_final.pth for all four environments

For the pruning experiments, run:
```
bash prune_pacs.sh
bash prune_officehome.sh
```
1. I parallelize pruning on all four test environments. You need to average the final results for each environment manually from the output logs.
2. Each code starts by generating the data-based DSS score from pretrained baseline. I also provide generated score by me in \*_\*_score.pkl for PACS and all four test environments.

For reproduction purpose, I also provide training logs in domainbed_miro_exp/train_output

## Acknowledgement
Some dataloading and evaluation code for MNIST-M is from:

https://github.com/NaJaeMin92/pytorch_DANN

Some dataloading and evaluation code for PACS and MIRO is from:

https://github.com/kakaobrain/miro


