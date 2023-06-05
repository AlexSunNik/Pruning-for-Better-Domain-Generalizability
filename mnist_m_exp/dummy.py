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

encoder = model.Extractor().cuda()
classifier = model.Classifier().cuda()

encoder(torch.tensor([0]))