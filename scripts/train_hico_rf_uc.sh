ulimit -n 4096
set -x
EXP_DIR=exps/hico/hoiclip

for i in 1
do
    swapon --show
    free -h
    export NCCL_P2P_LEVEL=NVL
    export OMP_NUM_THREADS=8
    python -m torch.distributed.launch \
            --nproc_per_node=1 \
            --use_env \
            main.py \
            --eval \
            --output_dir ${EXP_DIR} \
            --dataset_file hico \
            --hoi_path data/hico_20160224_det \
            --num_obj_classes 80 \
            --num_verb_classes 117 \
            --backbone resnet50 \
            --num_queries 64 \
            --dec_layers 3 \
            --epochs 5 \
            --lr_drop 60 \
            --use_nms_filter \
            --fix_clip \
            --batch_size 4 \
            --with_clip_label \
            --with_obj_clip_label \
            --gradient_accumulation_steps 1 \
            --num_workers 4 \
            --opt_sched "multiStep" \
            --dataset_root GEN \
            --model_name HOICLIP \
            --del_unseen \
            --zero_shot_type rare_first \
            --resume ${EXP_DIR}/checkpoint_last.pth \
            --verb_pth ./tmp/verb.pth \
            --training_free_enhancement_path \
            ./training_free_ehnahcement/
    sleep 120
done
