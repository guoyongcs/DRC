# train SwinIR-light-DRC
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1235 main_train_psnr_drc.py --opt options/swinir/train_swinir_sr_lightweight_drc.json --opt_teacher options/swinir/train_swinir_sr_classical.json --dist True

# test SwinIR-light-DRC for Set5
python main_test_swinir_drn.py --opt options/swinir/test_swinir_sr_lightweight_drc.json --task lightweight_sr_drc --folder_lq /data_root/dataset/sr/benchmark/Set5/LR_bicubic/X4 --folder_gt /data_root/dataset/sr/benchmark/Set5/HR --dataset Set5


# train SwinIR-DR
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1237 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical_dr.json  --dist True

# train SwinIR-DR on Set5
python main_test_swinir_dr.py --opt options/swinir/test_swinir_sr_classical_dr.json --task classical_sr_dr --folder_lq /data_root/dataset/sr/benchmark/Set5/LR_bicubic/X4 --folder_gt /data_root/dataset/sr/benchmark/Set5/HR --dataset Set5
