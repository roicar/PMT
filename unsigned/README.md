# Unsigned experiments

This directory contains the supplementary experiments on nonnegative data. It compares an Accum model trained on nonnegative integers using Cho-style unsigned encoding with the original Accum model trained on signed integers. Both models use the same numerical test instances with their respective encodings and position IDs. The polynomial multiplication experiments share the original Collected Product model.

## Directory structure

- `data_gen/` contains the scripts for generating integer accumulation and univariate polynomial multiplication datasets.
- `dataset/` stores the generated datasets as numbered `.pt` files.
- `data_utils/` contains argument parsing, tokenization, and data loader utilities.
- `data/` stores tokenized datasets and data loaders.
- `model_utils/` contains the training and inference code for these experiments.
- `model/` stores the Accum model trained in this directory.
- `log/` stores training and testing logs. The `cho/` and `signed/` subdirectories identify the Accum encoding.
- `run.py` is the entry point for training and testing.
- `preview_tokenized.py` displays raw samples or tokenized accumulation samples with their position IDs.
- `tests/` contains checks for the experiment protocol.

## Dataset generation

Run the commands from the PolyMUL root directory using the project's Python environment. Polynomial data generation calls Maple through `maple` on Linux or `cmaple` on Windows.

| Dataset | Task | Samples | Test samples |
| --- | --- | ---: | ---: |
| `C4o9` | Accumulation of 2 to 9 positive integers with up to 4 digits each | 1,000,000 | 200,000 |
| `v1d7C1` | Multiplication of two degree-7 univariate polynomials with coefficients from 0 to 9 | 100,000 | 20,000 |

```bash
python "unsigned/data_gen/accum/C{C}o{o}.py" --N 1000000 --C 4 --o 9
python "unsigned/data_gen/mul/v1d{d}C{C}.py" --N 100000 --d 7 --C 1
```

The scripts write to `unsigned/dataset/accum/C4o9/` and `unsigned/dataset/mul/v1d7C1/`. The data loader uses the first 70% of the samples for training, the next 10% for validation, and the last 20% for testing.

## Training

Train the Accum model with Cho-style encoding on `C4o9`.

```bash
python unsigned/run.py --dataset C4o9 --dataset_size 1000000 --batch_size 256 --accum --unsigned --train --log
```

When a run needs to build a data loader and `data.pt` is missing, the command first encodes the full dataset, saves `data.pt`, and exits. Run the same command again to start training or testing. Accumulation uses separate `cho/` and `signed/` caches under `unsigned/data/accum/C4o9_1000000/`. The two multiplication runs share the ExDs data loader.

The trained checkpoint is saved to `unsigned/model/accum/C4o9_1000000/cho/PE_pc_D_512_H_8_NE_0_DE_6.pth`.

## Testing

`--unsigned` selects Cho-style Accum encoding. Omitting it selects the original signed encoding for the same nonnegative data. The commands for the original model pass its checkpoint from the root `model/` directory. Each run with `--log` replaces any existing log for the same configuration.

### Integer accumulation

```bash
# Cho-style unsigned encoding
python unsigned/run.py --dataset C4o9 --dataset_size 1000000 --batch_size 256 --accum --unsigned --log

# Original signed encoding
python unsigned/run.py --dataset C4o9 --dataset_size 1000000 --batch_size 256 --accum --accum_checkpoint "model/accum/C4o9_1000000/PE_pc_D_512_H_8_NE_0_DE_6.pth" --log
```

### Polynomial multiplication

Both commands use the same Collected Product checkpoint and the `v1d7C1` test set. Each uses the corresponding Accum checkpoint from the integer accumulation test.

```bash
# Collected Product with Cho-style Accum
python unsigned/run.py --dataset v1d7C1 --dataset_size 100000 --batch_size 256 --tokenization ExDs --collect --accum --unsigned --collect_checkpoint "model/mul/v1d7C1_100000/ExDs/collect/PE_ape-l_D_512_H_8_NE_6_DE_6.pth" --log

# Collected Product with the original signed Accum
python unsigned/run.py --dataset v1d7C1 --dataset_size 100000 --batch_size 256 --tokenization ExDs --collect --accum --collect_checkpoint "model/mul/v1d7C1_100000/ExDs/collect/PE_ape-l_D_512_H_8_NE_6_DE_6.pth" --accum_checkpoint "model/accum/C4o9_1000000/PE_pc_D_512_H_8_NE_0_DE_6.pth" --log
```

## Test results

TA, SA, and MA denote token, sequence, and mathematical accuracy. The table reports the precision shown in the test logs.

| Dataset | Accum model | TA (%) | SA (%) | MA (%) |
| --- | --- | ---: | ---: | ---: |
| `C4o9` | Cho-style (unsigned) | 100.0000 | 100.0000 | 100.0000 |
| `C4o9` | Ours (signed) | 100.0000 | 100.0000 | 100.0000 |
| `v1d7C1` | Cho-style (unsigned) | 98.8279 | 97.9250 | 97.9250 |
| `v1d7C1` | Ours (signed) | 98.9452 | 98.0050 | 98.0050 |

See the full [accumulation logs](log/accum/C4o9_1000000/) and [polynomial multiplication logs](log/mul/v1d7C1_100000/ExDs/collect_and_accum/).

### Coded and annotated by Bin WANG
### Last modified time: 5 Nov 2026