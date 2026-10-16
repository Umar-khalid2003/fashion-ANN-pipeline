# change the first line for conflict
# 👗 Fashion-MNIST ANN — End-to-End ML Versioning Pipeline

> **End-to-End Machine Learning Versioning with Git, DVC & Google Drive**

A reproducible Machine Learning pipeline for classifying **Fashion-MNIST** images using a fully-connected **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

This project demonstrates an end-to-end ML workflow combining:

* 🧠 TensorFlow/Keras for model development
* 🔀 Git/GitHub for source-code version control
* 📦 DVC for dataset, model, and pipeline versioning
* ☁️ Google Drive as the DVC remote storage
* ⚙️ `params.yaml` for centralized hyperparameter management
* 🔄 `dvc.yaml` for reproducible pipeline orchestration
* 📊 Metrics and experiment comparison
* ⚔️ Git + DVC conflict simulation and resolution

The complete pipeline can be reproduced using a single command:

```bash
dvc repro
```

---

## 📌 Project Overview

### Problem Statement

The objective is to build a fully-connected Artificial Neural Network that classifies Fashion-MNIST images into one of **10 clothing categories**.

The dataset contains:

* **70,000 grayscale images**
* Image size: **28 × 28 pixels**
* **60,000 training images**
* **10,000 testing images**
* **10 classes**

The target is to achieve at least:

```text
≥ 85% test accuracy
```

The complete ML workflow is versioned and reproducible using Git and DVC.

---

## 🎯 Project Objectives

This project demonstrates the following capabilities:

### Git

* Repository initialization
* Branching and merging
* Commit history management
* Git log variants
* Git diff variants
* Stashing
* Rebasing
* Soft and hard reset
* File movement with `git mv`
* File removal with `git rm`
* Merge conflict resolution

### Machine Learning

* Fashion-MNIST data preparation
* Data preprocessing
* Pixel normalization
* Train/validation/test split
* ANN architecture design
* Model training
* Model evaluation
* Confusion matrix generation
* Metrics tracking

### DVC

* Dataset versioning
* Model versioning
* DVC pipeline creation
* Parameter tracking
* Pipeline reproduction
* Lock-file generation
* Experiment versioning
* Data conflict resolution

### Cloud Storage

Google Drive is configured as the DVC remote for storing large datasets and trained models.

---

# 🏗️ Project Architecture

```text
                         Fashion-MNIST
                              │
                              ▼
                       ┌──────────────┐
                       │  prepare.py  │
                       └──────┬───────┘
                              │
                              ▼
                         data/raw/
                              │
                              ▼
                     ┌─────────────────┐
                     │ preprocess.py   │
                     └────────┬────────┘
                              │
                              ▼
                      data/processed/
                              │
                              ▼
                       ┌─────────────┐
                       │   train.py  │
                       └──────┬──────┘
                              │
                              ▼
                       models/model.h5
                              │
                              ▼
                     ┌────────────────┐
                     │  evaluate.py   │
                     └───────┬────────┘
                             │
                             ▼
                         metrics.json
```

The pipeline is orchestrated by DVC:

```text
prepare
   ↓
preprocess
   ↓
train
   ↓
evaluate
```

---

# 📂 Project Structure

```text
fashion-ann-pipeline/
│
├── src/
│   ├── prepare.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── model.h5
│   └── history.csv
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── metrics.json
├── README.md
├── .gitignore
│
└── *.dvc
```

---

# 🔧 Technologies Used

| Technology         | Purpose                        |
| ------------------ | ------------------------------ |
| Python             | Programming language           |
| TensorFlow / Keras | ANN development                |
| Fashion-MNIST      | Dataset                        |
| NumPy              | Numerical processing           |
| Scikit-learn       | Evaluation utilities           |
| Matplotlib         | Visualization                  |
| PyYAML             | YAML configuration             |
| Git                | Source-code version control    |
| GitHub             | Remote Git repository          |
| DVC                | Data/model/pipeline versioning |
| Google Drive       | DVC remote storage             |

---

# ⚙️ Environment Setup

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fashion-ann-pipeline
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib
```

The assignment requires Python 3.9+ and these core packages.

---

# 🧠 Machine Learning Pipeline

## 1. Data Preparation

`src/prepare.py`

Downloads Fashion-MNIST using:

```python
tf.keras.datasets.fashion_mnist.load_data()
```

The raw arrays are saved under:

```text
data/raw/
```

Expected data:

```text
x_train
y_train
x_test
y_test
```

---

# 2. Data Preprocessing

`src/preprocess.py`

The preprocessing stage:

1. Loads raw data
2. Converts pixel values to floating point
3. Normalizes pixels to `[0, 1]`
4. Creates a validation split
5. Saves processed datasets

Output:

```text
data/processed/
```

---

# 3. ANN Training

`src/train.py`

The neural network follows the required architecture:

```text
Input Image
     │
     ▼
Flatten
     │
     ▼
Dense + ReLU
     │
     ▼
Dropout
     │
     ▼
Dense(10) + Softmax
     │
     ▼
Class Prediction
```

The model is compiled using:

```text
Optimizer:
Adam

Loss:
sparse_categorical_crossentropy
```

The trained model is saved as:

```text
models/model.h5
```

Training history is saved as:

```text
models/history.csv
```

---

# 4. Model Evaluation

`src/evaluate.py`

The evaluation stage:

* Loads the trained model
* Loads the processed test dataset
* Calculates test loss
* Calculates test accuracy
* Generates a confusion matrix
* Saves evaluation metrics

Output:

```text
metrics.json
```

Example:

```json
{
    "test_loss": 0.32,
    "test_accuracy": 0.87
}
```

> The actual values depend on the trained model and experiment configuration.

---

# ⚙️ Hyperparameter Management

All configurable training parameters are centralized in:

```text
params.yaml
```

Example:

```yaml
preprocess:
  test_size: 0.2
  seed: 42

train:
  dense_units: 128
  dropout_rate: 0.3
  learning_rate: 0.001
  epochs: 10
  batch_size: 32
```

This prevents hyperparameters from being hard-coded inside the Python scripts.

The Python scripts load the configuration using:

```python
yaml.safe_load()
```

This makes experiments easier to reproduce and compare.

---

# 🔄 DVC Pipeline

The complete pipeline is defined in:

```text
dvc.yaml
```

The pipeline contains four stages:

```text
prepare
   ↓
preprocess
   ↓
train
   ↓
evaluate
```

Each stage defines its:

* Command
* Dependencies
* Outputs
* Parameters
* Metrics where applicable

---

# ▶️ Running the Pipeline

Instead of manually executing every script, the entire ML workflow can be reproduced with:

```bash
dvc repro
```

DVC determines which stages need to execute based on:

* Source-code changes
* Dataset changes
* Parameter changes
* Dependencies
* Previous pipeline state

---

# 🔐 DVC Lock File

After running:

```bash
dvc repro
```

DVC generates:

```text
dvc.lock
```

The lock file records the exact state of the pipeline dependencies and outputs.

It should be committed to Git:

```bash
git add dvc.lock
git commit -m "Update DVC lock file"
```

This allows the pipeline state to be reproduced consistently.

---

# ☁️ Google Drive as DVC Remote

Google Drive is used as the remote storage location for large DVC-tracked artifacts.

Initialize DVC:

```bash
dvc init
```

Add the Google Drive remote:

```bash
dvc remote add -d gdrive_storage gdrive://<GOOGLE_DRIVE_FOLDER_ID>
```

Then authenticate through Google when performing the first push.

---

# 📦 Tracking Data and Models with DVC

Track the major ML artifacts:

```bash
dvc add data/raw
dvc add data/processed
dvc add models
```

Then commit the generated DVC pointer files:

```bash
git add data/raw.dvc
git add data/processed.dvc
git add models.dvc
git add .gitignore
git commit -m "Track datasets and models with DVC"
```

Push the actual artifacts to Google Drive:

```bash
dvc push
```

---

# 🔀 Git + DVC Versioning Model

This project uses Git and DVC together.

```text
                    GitHub
                      │
             ┌────────┴────────┐
             │                 │
          Source Code       DVC Files
             │                 │
             │            dvc.yaml
             │            dvc.lock
             │            *.dvc
             │
             └────────┬────────┘
                      │
                      ▼
                     DVC
                      │
                      ▼
                 Google Drive
                      │
             ┌────────┴────────┐
             │                 │
           Datasets          Models
```

### Git manages

```text
Python source code
README.md
params.yaml
dvc.yaml
dvc.lock
.dvc pointer files
```

### DVC manages

```text
Raw datasets
Processed datasets
Trained models
Large ML artifacts
```

### Google Drive stores

```text
Actual DVC-tracked artifacts
```

---

# 🧪 Experiment Versioning

One of the main goals is to demonstrate that changing a parameter causes only the necessary pipeline stages to rerun.

For example, change:

```yaml
dense_units: 128
```

to:

```yaml
dense_units: 256
```

Then execute:

```bash
dvc repro
```

Because the training configuration changed, the training-dependent stages are reproduced while unaffected stages can remain unchanged.

This demonstrates DVC's pipeline dependency tracking.

---

# 🏷️ Experiment Tags

The project uses Git tags to identify experiment versions.

For example:

```bash
git tag v1
```

and:

```bash
git tag v2
```

This allows the code/configuration state associated with different experiments to be identified.

---

# 📊 Metrics Comparison

The final report includes a comparison between the experiment versions.

Example format:

| Version | Dense Units | Test Accuracy |
| ------- | ----------: | ------------: |
| v1      |         128 |        XX.XX% |
| v2      |         256 |        XX.XX% |

Replace the values with the actual metrics generated by the experiments.

---

# 🌿 Git Workflow

The project uses a development branch:

```text
main
  │
  └── dev
```

Feature development takes place on `dev`.

Example workflow:

```bash
git checkout -b dev

git add .
git commit -m "Add preprocessing pipeline"

git add .
git commit -m "Add ANN training pipeline"

git add .
git commit -m "Add evaluation pipeline"
```

---

# 🧰 Git Commands Demonstrated

The project demonstrates the following Git functionality.

### Repository

```bash
git init
```

### Branching

```bash
git checkout -b dev
```

### Logging

```bash
git log --oneline --graph --all
git log --stat -3
git log -p -1
git log main..dev
```

### Differences

```bash
git diff
git diff --staged
git diff main..dev
git diff main...dev
```

### Stashing

```bash
git stash
git stash list
git stash pop
```

### Rebasing

```bash
git rebase main
```

### Reset

```bash
git reset --soft HEAD~1
git reset --hard HEAD~1
```

### File management

```bash
git mv
git rm
```

---

# ⚔️ Conflict Resolution

The project also simulates a two-contributor environment.

Two branches independently modify:

```text
src/preprocess.py
```

and regenerate:

```text
data/processed/
```

This creates two types of conflicts:

### 1. Git Code Conflict

Different changes to:

```text
src/preprocess.py
```

must be manually reconciled.

### 2. DVC Data Conflict

Different branches contain different DVC pointer hashes for:

```text
data/processed/
```

The appropriate data version is selected or regenerated, followed by:

```bash
dvc checkout
```

to synchronize the working directory.

Finally:

```bash
dvc status
dvc repro
```

are used to verify that the resolved project is clean and reproducible.

---

# 🔁 Complete Reproduction Workflow

A fresh environment can reproduce the ML pipeline using:

```bash
dvc pull
```

followed by:

```bash
dvc repro
```

The conceptual workflow is:

```text
Clone Repository
       ↓
Install Dependencies
       ↓
DVC Pull
       ↓
Download/Restore DVC Artifacts
       ↓
DVC Repro
       ↓
Prepare
       ↓
Preprocess
       ↓
Train
       ↓
Evaluate
       ↓
Metrics
```

---

# 📈 Expected Outcome

The final system should produce:

```text
Raw Dataset
     ↓
Processed Dataset
     ↓
Trained ANN
     ↓
Evaluation Metrics
     ↓
Confusion Matrix
```

with a target test accuracy of:

```text
≥ 85%
```

The exact final accuracy depends on the hyperparameters and training configuration used.

---

# 📝 Assignment Deliverables

The completed project provides the required materials:

### 1. GitHub Repository

Contains:

* Complete source code
* Git history
* `params.yaml`
* `dvc.yaml`
* `dvc.lock`
* DVC pointer files
* README

### 2. Google Drive DVC Remote

Contains the DVC-managed data/model artifacts.

### 3. PDF Report

The report documents:

* Git command demonstrations
* Screenshots
* DVC configuration
* Google Drive remote
* `dvc.yaml`
* `params.yaml`
* `dvc repro` execution
* Hyperparameter experiment
* Git conflict
* DVC conflict
* Conflict resolution
* v1 vs v2 metrics

---

# 🎓 Learning Outcomes

After completing this project, the following workflow should be understood:

```text
                 MACHINE LEARNING
                        │
                        ▼
                 Build the Model
                        │
                        ▼
                 Version the Code
                        │
                        ▼
                Version the Data
                        │
                        ▼
              Define the Pipeline
                        │
                        ▼
             Reproduce Experiments
                        │
                        ▼
              Track Model Versions
                        │
                        ▼
              Resolve Data Conflicts
                        │
                        ▼
             Reproduce Final System
```

The project therefore demonstrates the transition from a simple ML experiment to a **version-controlled, reproducible ML workflow**.

---

# 👨‍💻 Author

**Muhammad Umar Khalid**

Data Science Student
FAST-NUCES

---

# 📜 Assignment Context

This repository was developed as part of **Assignment 2 — End-to-End ML Versioning with Git, DVC & Google Drive**.

The assignment specifies the Fashion-MNIST ANN task, Git exercises, DVC/Google Drive setup, pipeline requirements, experiment versioning, and conflict-resolution workflow.
