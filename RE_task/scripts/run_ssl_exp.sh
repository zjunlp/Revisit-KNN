
TASK=semeval    # [semeval, tacrev]
lr=3e-5         # semeval: 3e-5 (pt) / 8e-5 (ft)

temp=10         # [0.1, 1, 10]
alpha=0.001     # [0.0001, 0.001, 0.01, 0.1]
knn_lambda=0.5  # [0.1 : .1 : 0.9]

for seed in 1 2 3
do
echo $seed

CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=25  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir dataset/$TASK \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --use_template_words 0 \
    --init_type_words 0 \
    --output_dir ckpt/$TASK/ssl/pt-knn-train \
    --ssl \
    --train_with_knn \
    --knn_infer \
    --temp $temp \
    --alpha $alpha \
    --knn_topk 64 \
    --knn_lambda $knn_lambda
done