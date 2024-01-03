#### Search for 4x DRN (please add more scripts for 8x under diverse pruning rates)
CUDA_VISIBLE_DEVICES=0 python main.py --n_GPUs 1 --search --scale 4 --template DRNS --pre_train ./pretrained_models/DRNS4x.pt --pre_train_dual ./pretrained_models/DRNS4x_dual_model.pt --batch_size 16 --patch_size 192 --epochs 100 --ext img --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --pruning_rate 0.3 --test_every 50 --save ../search_experiment/search_x4p03

#### Pruning for 4x DRN
CUDA_VISIBLE_DEVICES=4 python main.py --pruning --scale 4 --model DRN --template DRNS --pre_train ./pretrained_models/DRNS4x.pt --pre_train_dual ./pretrained_models/DRNS4x_dual_model.pt --save ../prune_experiment/prune_searchx403 --test_every 10 --ext img --patch_size 192 --batch_size 16 --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --configs_path ./pretrained_models/search_x4p03.txt

CUDA_VISIBLE_DEVICES=3 python main.py --pruning --scale 4 --model DRN --template DRNS --pre_train ./pretrained_models/DRNS4x.pt --pre_train_dual ./pretrained_models/DRNS4x_dual_model.pt --save ../prune_experiment/prune_searchx405 --test_every 10 --ext img --patch_size 192 --batch_size 16 --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --configs_path ./pretrained_models/search_x4p05.txt

CUDA_VISIBLE_DEVICES=2 python main.py --pruning --scale 4 --model DRN --template DRNS --pre_train ./pretrained_models/DRNS4x.pt --pre_train_dual ./pretrained_models/DRNS4x_dual_model.pt --save ../prune_experiment/prune_searchx407 --test_every 10 --ext img --patch_size 192 --batch_size 16 --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --configs_path ./pretrained_models/search_x4p07.txt

#### Pruning for 8x DRN 
CUDA_VISIBLE_DEVICES=5 python main.py --pruning --scale 8 --model DRN --template DRNS --pre_train ./pretrained_models/DRNS8x.pt --pre_train_dual ./pretrained_models/DRNS8x_dual_model.pt --save ../prune_experiment/prune_searchx803 --test_every 10 --ext img --patch_size 192 --batch_size 16 --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --configs_path ./pretrained_models/search_x8p03.txt

CUDA_VISIBLE_DEVICES=6 python main.py --pruning --scale 8 --model DRN --template DRNS --pre_train ./pretrained_models/DRNS8x.pt --pre_train_dual ./pretrained_models/DRNS8x_dual_model.pt --save ../prune_experiment/prune_searchx805 --test_every 10 --ext img --patch_size 192 --batch_size 16 --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --configs_path ./pretrained_models/search_x8p05.txt

CUDA_VISIBLE_DEVICES=7 python main.py --pruning --scale 8 --model DRN --template DRNS --pre_train ./pretrained_models/DRNS8x.pt --pre_train_dual ./pretrained_models/DRNS8x_dual_model.pt --save ../prune_experiment/prune_searchx807 --test_every 10 --ext img --patch_size 192 --batch_size 16 --data_dir /data_root/dataset/sr/ --data_train DIV2KS --data_test Set5 --configs_path ./pretrained_models/search_x8p07.txt




#### finetune pruned 30% drns for x4 SR (please add more scripts for 8x under diverse pruning rates)
python main.py --data_dir /data_root/sr/ --data_train DF2K --scale 4 --n_GPUs 4 --template DRNL --ft_model SDRN --finetune --configs_path ./pretrained_models/search_x4p03.txt --save ../prune_experiment/finetune_prune_searchx407 --pre_train ./pretrained_models/DRNL4x.pt --pre_train_dual ./pretrained_models/DRNL4x_dual_model.pt --pruned_model ../prune_experiment/prune_searchx403/checkpoint/pruned_model.pth --pruned_dual_model ../pretrained_models/DRNS4x_dual_model.pt

