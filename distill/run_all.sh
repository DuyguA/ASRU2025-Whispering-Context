#!/bin/bash

# Base parameters
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
EPOCHS=1
LR=5e-5
TEMPERATURE=2.0
LOSS_TYPE="CLS"
TEACHER_DIM=1024
WSIZE=64

# Values to iterate over
VALUES=(0.0 0.0005 0.001 0.002 0.005 0.01)

# Iterate over combinations of alpha and beta
for ALPHA in "${VALUES[@]}"; do
  for BETA in "${VALUES[@]}"; do
    # Skip the alpha=0 and beta=0 combination
    if [[ "$ALPHA" == "0.0" && "$BETA" == "0.0" ]]; then
      continue
    fi

    # Set the output directory
    OUTPUT_DIR="whisper-${ALPHA}-${BETA}"

    # Run the training script
    python3 -u trainer.py --train_batch_size=$TRAIN_BATCH_SIZE \
                          --val_batch_size=$VAL_BATCH_SIZE \
                          --epochs=$EPOCHS \
                          --lr=$LR \
                          --teacher_dim=$TEACHER_DIM \
                          --temperature=$TEMPERATURE \
                          --loss_type=$LOSS_TYPE \
                          --alpha=$ALPHA \
                          --beta=$BETA \
		          --window_size=$WSIZE \
		          --output_dir=$OUTPUT_DIR
  done
done

