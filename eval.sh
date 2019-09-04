run_id=14
z_A=10
z_B=10
z_S=1
gpu=0
ckpt=100000


lamkl=4
beta1=15
beta2=15
beta3=0



CUDA_VISIBLE_DEVICES=${gpu} \
nohup python -u main.py \
--output_save_iter 1 --run_id ${run_id} \
--zA_dim ${z_A} --zB_dim ${z_B} --zS_dim ${z_S} \
--beta1 ${beta1} --beta2 ${beta2} --beta3 ${beta3} \
--ckpt_load_iter ${ckpt} \
--dset_dir ../ --image_size 128 --dataset digit12 --num_workers 2 --batch_size 64 \
--print_iter 50 \
--ckpt_save_iter 10000 \
--max_iter 100001  --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lamkl ${lamkl} \
--eval_metrics --eval_metrics_iter 1 \
--viz_ll_iter 10 --viz_la_iter 50 --viz_port 8002 \
> ./log/eval${run_id}.${z_A}.${z_B}.${z_S}.txt &
# 8001=cbb17
