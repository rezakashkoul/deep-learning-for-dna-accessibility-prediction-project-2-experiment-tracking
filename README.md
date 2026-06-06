# Project 2: Experiment Tracking and Transfer Learning for DNA Accessibility Prediction

This project extends the Project 1 DNA accessibility prediction pipeline into a more complete and reproducible experimentation workflow using PyTorch, TensorBoard, automated hyperparameter search, advanced evaluation metrics, model checkpointing, and frozen-backbone transfer learning.

## Overview

In Project 1, a strict 5-layer 1D CNN was implemented for DNA accessibility prediction using DNase-seq sequence data.
In Project 2, the same core model and dataset pipeline were reused and extended with the following features:

- TensorBoard integration for experiment tracking
- Automated hyperparameter grid search
- Advanced evaluation using confusion matrix, precision, recall, and F1-score
- Save-best-model checkpointing
- Frozen-backbone transfer learning on a different organ dataset

## Project Structure

project2/
├── final_code/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── grid_search.py
│   ├── frozen_backbone_heart.py
│   ├── grid_results/
│   ├── frozen_heart_results_reinit/
│   ├── runs/
│   ├── runs_grid/
│   └── runs_frozen_heart/
├── project_harmonized_174/
│   ├── brain/
│   ├── blood/
│   └── heart/
├── project_selected/
├── figures/
└── final_dataset/

## Dataset Summary

### Brain dataset

The main Brain dataset contains five experiments:

- ENCFF068WQO
- ENCFF109HOA
- ENCFF263JBT
- ENCFF738LML
- ENCFF750GSJ

From each Brain experiment:

- 5,000 positive samples were selected
- 5,000 negative samples were selected

Final Brain dataset size:

- 25,000 positive
- 25,000 negative
- 50,000 total samples

The Brain dataset was split into:

- 70% training
- 15% validation
- 15% test

### Heart dataset

For transfer learning, a different organ was used:

- ENCFF586UTZ (Heart)

The Heart dataset was harmonized to sequence length 174 and split into:

- 70% training
- 30% test

## DNA Encoding

Each DNA sequence was one-hot encoded as:

- A = [1, 0, 0, 0]
- C = [0, 1, 0, 0]
- G = [0, 0, 1, 0]
- T = [0, 0, 0, 1]

Each sequence of length 174 was converted from shape:

- 174 × 4

to:

- 4 × 174

for compatibility with Conv1D.

## Model Architecture

The model in `model.py` is a strict 5-layer 1D CNN:

1. Conv1D + ReLU + MaxPool
2. Conv1D + ReLU + MaxPool
3. Conv1D + ReLU + MaxPool
4. Dense Layer
5. Output Layer with Sigmoid activation

Dropout is used in the first fully connected layer to reduce overfitting.

## Project 2 Tasks

### Task 1: TensorBoard Integration

The updated training pipeline logs:

- Training Loss
- Test Loss
- Test Accuracy
- Model graph with `writer.add_graph()`

### Task 2: Automated Hyperparameter Grid Search

The grid search explores:

- Learning rate: 0.001, 0.0001
- Batch size: 16, 64
- Kernel size: 5, 11

Total runs:

- 8 runs

Each run saves:

- TensorBoard logs
- Best model checkpoint
- Accuracy curve
- Loss curve
- Confusion matrix
- `metrics.txt`

### Task 3: Advanced Evaluation

The pipeline computes:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score

The confusion matrix is also logged to TensorBoard using `writer.add_figure()`.

### Task 4: Model Checkpointing

The best model is saved whenever test accuracy improves.

### Task 5: Frozen-Backbone Transfer Learning

The best pretrained Brain model is loaded, the convolutional backbone is frozen, the classifier is reinitialized, and only the classifier layers are fine-tuned on the Heart dataset for 5 epochs.

## Best Grid Search Result

The best grid-search configuration was:

- Learning Rate = 0.001
- Batch Size = 16
- Kernel Size = 11

Performance:

- Accuracy = 0.7677
- Precision = 0.7187
- Recall = 0.7877
- F1-score = 0.7516

Best checkpoint:

- `grid_results/grid_run_2_lr_0.001_bs_16_ks_11/best_model.pth`

## Frozen Backbone Transfer Learning Result

Final Heart transfer-learning result with frozen backbone and reinitialized classifier:

- Test Loss = 0.5765
- Accuracy = 0.6935
- Precision = 0.6901
- Recall = 0.7070
- F1-score = 0.6984

In this experiment:

- `conv1`, `conv2`, and `conv3` were frozen
- `fc1` and `fc2` were reinitialized
- only classifier layers were updated

## Main Files

### `dataset.py`

Responsible for:

- one-hot encoding
- custom PyTorch dataset
- balanced Brain dataset construction
- Brain split creation
- Blood dataset loading
- Heart dataset loading

### `model.py`

Defines the 5-layer 1D CNN used in both Project 1 and Project 2.

### `train.py`

Used for:

- TensorBoard logging
- test-based metric tracking
- checkpointing
- standard experiment execution

### `grid_search.py`

Runs all hyperparameter combinations automatically.

### `frozen_backbone_heart.py`

Runs frozen-backbone transfer learning on the Heart dataset using the best pretrained Brain model.

## Example Commands

### Run a standard experiment

python3 train.py --experiment 3

### Run grid search

python3 grid_search.py

### Run frozen-backbone transfer learning

python3 frozen_backbone_heart.py \
  --best_model_path grid_results/grid_run_2_lr_0.001_bs_16_ks_11/best_model.pth \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --results_root frozen_heart_results_reinit

## TensorBoard

### Standard runs

tensorboard --logdir runs --port 6006

### Grid search runs

tensorboard --logdir runs_grid --port 6006

### Frozen backbone runs

tensorboard --logdir runs_frozen_heart --port 6006

## Environment

Experiments were executed on:

- Server account: `kashkoul@pc507.emulab.net`
- Hostname: `node0.dnaprediction.siue-cs590-490.emulab.net`
- OS: Ubuntu 22.04.2 LTS
- CPU: Intel(R) Xeon(R) CPU E5530 @ 2.40GHz
- CPU count: 8
- Memory: 11 GiB RAM

## Conclusion

This project extended the original DNA accessibility prediction pipeline into a reproducible experiment-tracking system. It added visualization, hyperparameter search, better evaluation, checkpointing, and transfer learning while preserving the Project 1 model foundation.
