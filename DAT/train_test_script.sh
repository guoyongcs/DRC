# train SwinIR-DR
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="./:${PYTHONPATH}" python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_x4_dr.yml --launcher pytorch --auto_resume

# test SwinIR-DR on Set5
PYTHONPATH="./:${PYTHONPATH}" python basicsr/test.py -opt options/Test/test_DAT_x4_dr.yml
