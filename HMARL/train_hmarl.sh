#CUDA_VISIBLE_DEVICES=6,8,9 python3 train_one_stage.py -opt configs/one_stage/train_config.yml --auto_resume
CUDA_VISIBLE_DEVICES=0,1,2 python3 train_one_stage.py -opt configs/one_stage/train_config.yml --auto_resume