#!/bin/bash
### Integer accumulation
## dataset: C4o9, size: 1,000,000
python run.py --dataset C4o9 --dataset_size 1000000 --batch_size 512 --train --accum --log
python run.py --dataset C4o9 --dataset_size 1000000 --batch_size 512 --valid --accum --log # Inference on validation dataset
python run.py --dataset C4o9 --dataset_size 1000000 --batch_size 512 --accum --log # Inference on testing dataset
### Multiplication
## dataset: [v1d1C1, v2t2E1C1, v3t3E1C1], size: 100,000
python run.py --train --batch_size 512 --tokenization yyy --log
# yyy ranges from [ImUs, ImDs, ExUs, ExDs]
## dataset: [v1d3C1, v1d7C1], size: 100,000
# Directly learn product
python run.py --dataset v1d3C1 --batch_size 256 --tokenization yyy --log
# Learn collected product first and polynomial simplification next
python run.py --dataset v1d3C1 --batch_size 256 --tokenization yyy --train --log --collect # Training model for learning collected product
python run.py --dataset v1d3C1 --batch_size 256 --tokenization yyy --train --log --combine # Training model for learning polynomial simplification
python run.py --dataset v1d3C1 --batch_size 256 --tokenization yyy --valid --log --collect --combine # Inference (take validation dataset as example)
# Multi-step learning strategy
python run.py --dataset v1d3C1 --batch_size 256 --tokenization zzz --valid --log --collect --accum # Inference (take validation dataset as example)
# zzz ranges from [ImDs, ExDs]
# Long-sequence multiplication (dataset: v1d15C1)
python run.py --dataset v1d15C1 --batch_size 64 --train --log --collect # Training model for learning collected product of each pair of sub-polynomials
python run.py --dataset v1d15C1 --batch_size 64 --train --log --sum # Training model for learning collected sum of incremental addition of sub-product to the final product
python run.py --dataset v1d15C1 --batch_size 64 --valid --log --collect --sum --accum # Whole inference process
### Pseudodivision
## Directly learn remainder (dataset: [v1d3C1, v1d7C1, _v1D3C1L3, _v1D7C1L3, v2t2E1C1L3, v3t3E1C1L3, v2t2e4F7], size: 100,000)
python run.py --task prem --dataset v1d3C1 --batch_size 256 --valid --log
## Finetune multiplication model (dataset: [v1d3C1, v1d7C1, v3t3E1C1L3], size: 100,000)
python run.py --task prem --dataset v1d3C1 --batch_size 256 --valid --log --finetune --no_freeze # `--no_freeze` indicating not freezing decoder layers in multiplication model
## 1-step Scratchpad (dataset: [v1d3C1, v1d7C1, _v1D3C1L3, _v1D7C1L3, v2t2E1C1L3], size: 100,000)
python run.py --task prem --dataset _v1D3C1L3 --batch_size 256 --valid --log --prem_step 1
## N-step Scratchpad (dataset: [_v1D3C1L3, _v1D7C1L3, v2t2E1C1L3], size: 100,000)
python run.py --task prem --dataset _v1D7C1L3 --batch_size 256 --valid --log --prem_step N
## Detailed Scratchpad (dataset: [v1d7C1, _v1D3C1L3, _v1D7C1L3, v2t2E1C1L3], size: 100,000)
python run.py --task prem --dataset v2t2E1C1L3 --batch_size 256 --valid --log --prem_step N --detailed_spa
## Finetune model trained from integer-coefficient prem (dataset: [v2t2E1F7, v3t3E1F7], size: 100,000)
python run.py --task prem --dataset v3t3E1F7 --batch_size 256 --valid --log --finetune --no_freeze
### Characteristic set
## Directly learn characteristic set (dataset: [v2t2e2C1L3, v3t3e2C1L3, v2t2e2F7], size: 100,000)
python run.py --task charset --dataset v2t2e2C1L3 --batch_size 256 --valid --log
## Directly learn characteristic set (dataset: [v3t3e2C1L3, v2t2e4F7], size: 100,000)
SEQ_LEN=1024 python run.py --task charset --dataset v2t2e4F7 --batch_size 256 --valid --log
## Directly learn characteristic set (dataset: v3t3e2F7, size: 100,000)
SEQ_LEN=1280 python run.py --task charset --dataset v3t3e2F7 --batch_size 256 --valid --log
## Finetune prem model (dataset: [v3t3e2C1L3, v2t2e2F7], size: 100,000)
SEQ_LEN=1024 python run.py --task charset --dataset v2t2e2F7 --batch_size 256 --valid --log --finetune --no_freeze
## Finetune prem model (dataset: v3t3e2F7, size: 100,000)
SEQ_LEN=1280 python run.py --task charset --dataset v3t3e2F7 --batch_size 256 --valid --log --finetune --no_freeze
## Finetune prem model (dataset: v3t3E1F7S2, size: 40,000)
SEQ_LEN=1920 python run.py --task charset --dataset_size 40000 --batch_size 64 --dataset v3t3E1F7S2 --valid --log --finetune --no_freeze
