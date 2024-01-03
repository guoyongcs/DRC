# search
python search.py -opt options/setting2/train/search_setting2_x4_drc.yml --save_dir ./experiment/DCLS_experiments/search_p03


# prune
python pruning.py -opt options/setting2/train/pruning_setting2_x4_drc.yml --save_dir ./experiment/DCLS_experiments/pruning_p03

# finetune
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 train.py -opt=options/setting2/train/train_setting2_x4_drc.yml --launcher pytorch --train_dataroot /data_root/DCLS_Data/sr --exp_root ./DCLS

# test
python test.py -opt options/setting2/test/test_setting2_x4_drc.yml
