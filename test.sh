# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.15.0
cd /home/xyy/ESA_backup_0720/ESANet-main
# CUDA_VISIBLE_DEVICES=0 python eval.py --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_2_1.5/codestream/396-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=1 python eval.py --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_3_2.5/codestream/399-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=2 python eval.py --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_4_3.5/codestream/397-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=3 python eval.py --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_5_4.5/codestream/399-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=0 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_2_1.5/codestream/396-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=1 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_3_2.5/codestream/399-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_4_3.5/codestream/397-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=3 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse_5_4.5/codestream/399-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=0 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_Mse_2_1/codestream/401-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=1 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_Mse_3_2/codestream/401-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_Mse_4_3/codestream/401-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=3 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_Mse_5_4/codestream/401-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=0 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse-1.5_2_0.5/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=1 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse-1.5_3_1.5/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse-1.5_4_2.5/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=3 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_mse-1.5_5_3.5/codestream/*rep* 

# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_depth_ELIC_master-60_1/codestream/401-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_depth_ELIC_master-60_2/codestream/401-padding-replicate0 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_depth_ELIC_master-60_3/codestream/401-padding-replicate0 &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_depth_ELIC_master-60_4/codestream/401-padding-replicate0 

# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_ssim_2_2/codestream/*ref* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_ssim_3_3/codestream/*ref* 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_ssim_4_4/codestream/*ref* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_ssim_5_5/codestream/*ref*

# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_l1_2_2/codestream/*ref* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_l1_3_3/codestream/*ref* 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_l1_4_4/codestream/*ref* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_l1_5_5/codestream/*ref*

# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_2_2/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_3_3/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_4_4/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_5_5/codestream/*rep*

# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_2_1.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_3_2.5/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_4_3.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_5_4.5/codestream/*rep*

# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_2_2.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_3_3.5/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=2 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_4_4.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_5_5.5/codestream/*rep*

# CUDA_VISIBLE_DEVICES=5 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_cat_2/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=6 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_cat_3/codestream/*rep* 
# CUDA_VISIBLE_DEVICES=5 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_cat_4/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=6 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_cat_5/codestream/*rep*

# CUDA_VISIBLE_DEVICES=5 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_STF_united_add_2_1.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=5 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_STF_united_add_3_2.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=5 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_STF_united_add_4_3.5/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=5 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_STF_united_add_5_4.5/codestream/*rep* &

# CUDA_VISIBLE_DEVICES=6 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_heavy_2_2/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=7 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_heavy_3_3/codestream/*rep*
# CUDA_VISIBLE_DEVICES=6 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_heavy_4_4/codestream/*rep* &
# CUDA_VISIBLE_DEVICES=7 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/nyuv2_ELIC_united_heavy_5_5/codestream/*rep*

CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/sunrgbd_ELIC_cat_2/codestream/*rep* --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth &
CUDA_VISIBLE_DEVICES=7 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/sunrgbd_ELIC_cat_3/codestream/*rep* --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth 
CUDA_VISIBLE_DEVICES=4 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/sunrgbd_ELIC_cat_4/codestream/*rep* --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth  &
CUDA_VISIBLE_DEVICES=7 python eval.py --rec_data_dir /home/xyy/ELIC/experiments/sunrgbd_ELIC_cat_5/codestream/*rep* --dataset sunrgbd --dataset_dir /data/xyy/sunrgbd/ --ckpt_path /home/xyy/ESA_backup_0720/ESANet-main/trained_models/sunrgbd/r34_NBt1D.pth 


cd ..