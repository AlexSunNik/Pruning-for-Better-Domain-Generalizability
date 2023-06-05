import torch
import train
import mnist
import mnistm
import model
from utils import get_free_gpu
import argparse

save_name = 'mnist'


def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument('--mode', type=str, required=True, default='source')
    args = parser.parse_args()
    mode = args.mode

    if torch.cuda.is_available():
        # get_free_gpu()
        # print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()

        if mode == "source":
            train.source_only(encoder, classifier, source_train_loader, target_train_loader, save_name)
        else: 
            print("Unrecognized Name")
            exit()
    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    main()
