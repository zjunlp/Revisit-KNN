# Required environment variables:
# TYPE: finetune / prompt
# TASK: SST-5 / trec / MNLI / QNLI / 
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 /  42 / 100)
# MODEL: pre-trained model name (roberta-*), see Transformers model list

# Number of training instances per label

K=16            # 16-shot
TAG=exp
BS=8
TYPE="prompt"   # finetune or prompt
LR=1e-5         # [1e-5, 2e-5, 5e-5]
MODEL="roberta-large"

# Training steps
MAX_STEP=2000

# Validation steps
EVAL_STEP=200

# Gradient accumulation steps
REAL_BS=2   # [2, 4, 8]
GS=$(expr $BS / $REAL_BS)

# knn hyper parameters
topk=64     # [16, 32, 64, 128, 256, 512]
lambda=0.8  # [0.1 : .1 : 0.9]
temp=10     # [0.01, 0.1, 1.0, 10]

TASK=sst-5
for SEED in 13 42 100
do
echo $SEED

# Task specific parameters
# The default length is 128.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""
case $TASK in
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    trec)
        TEMPLATE="*cls**mask*:*+sent_0**sep+*"
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110 --double_demo"
        ;;
esac

# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM
DATA_DIR=data/original_data/glue/$TASK
OUTPUT_DIR=ckpt/pt_ssl_infer/$TASK/$REAL_BS-$LR

CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --do_predict \
  --evaluate_during_training \
  --model_name_or_path $MODEL \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length 128 \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir $OUTPUT_DIR \
  --seed $SEED \
  --tag $TAG \
  --template $TEMPLATE \
  --mapping $MAPPING \
  $TASK_EXTRA \
  $1 \
  --ssl \
  --temp $temp \
  --knn_infer \
  --knn_topk $topk \
  --knn_lambda $lambda

done
