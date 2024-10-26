gpu=$1
model=$2
bert_dir=$3
output_dir=$4
add1=$5
add2=$6
add3=$7

# ./evaluation_pipeline.sh 0 bert bert-base-uncased save/BERT


# DST
CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --my_model=BeliefTracker \
    --model_type=${model} \
    --dataset='["multiwoz"]' \
    --task_name="dst" \
    --earlystop="joint_acc" \
    --output_dir=${output_dir}/DST/MWOZ \
    --do_train \
    --task=dst \
    --example_type=turn \
    --model_name_or_path=${bert_dir} \
    --batch_size=6 --eval_batch_size=6 \
    --usr_token=[USR] --sys_token=[SYS] \
    --eval_by_step=4000 \
    $add1 $add2 $add3
