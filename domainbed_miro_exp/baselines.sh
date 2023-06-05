CUDA_VISIBLE_DEVICES=0 python3 train_all.py PACS --data_dir ../domain_bed/ --algorithm MIRO --dataset PACS --lr 3e-5 --resnet_dropout 0.0 --weight_decay 0.0 --ld 0.01
CUDA_VISIBLE_DEVICES=1 python3 train_all.py VLCS --data_dir ../domain_bed/ --algorithm MIRO --dataset VLCS --lr 1e-5 --resnet_dropout 0.5 --weight_decay 1e-6 --ld 0.01 --model_save 3500
CUDA_VISIBLE_DEVICES=2 python3 train_all.py OfficeHome --data_dir ../domain_bed/ --algorithm MIRO --dataset OfficeHome --lr 3e-5 --resnet_dropout 0.1 --weight_decay 1e-6 --ld 0.1 --model_save 3500
CUDA_VISIBLE_DEVICES=3 python3 train_all.py TerraIncognita --data_dir ../domain_bed/ --algorithm MIRO --dataset TerraIncognita --lr 3e-5 --resnet_dropout 0.0 --weight_decay 1e-4 --ld 0.1 --model_save 3500