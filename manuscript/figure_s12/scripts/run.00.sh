#!/bin/bash

RUNFILE="/home/mvinyard/experiments/pancreas/v4/run_scdiffeq.pancreas_cytotrace_time.py"

${RUNFILE} --mu_hidden="[512,512]" \
             --sigma_hidden="[32,32]" \
             --velocity_ratio_target=25e-1 \
             --velocity_ratio_enforce=100 \
             --train_epochs=2500 \
             --train_lr=1e-4 \
             --train_step_size=1500 \
             --n_seeds=5 \
             --swa_lrs=1e-5 \
             --mu_dropout=0 \
             --sigma_dropout=0 \
             --batch_size=512 \
             --potential_type='fixed' \
             --coef_g=1
             
             
             