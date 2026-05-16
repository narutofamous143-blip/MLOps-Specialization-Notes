# MLOps Tools for Freshers — Company-Level Practical Guide
### What You'll Actually Use at Work, How to Use It, and the Tricks Senior Engineers Know

> **How to read this guide**: Each tool section follows the same structure — what it is, when you'll touch it at work, full working code, and a "🔥 Important Tricks" block containing the non-obvious patterns that distinguish juniors from seniors. These tricks come directly from real production environments.

---

## Table of Contents

1. [MLflow — Experiment Tracking](#1-mlflow--experiment-tracking)
2. [Weights & Biases (W&B) — Experiment Tracking Alt.](#2-weights--biases-wb)
3. [DVC — Data & Pipeline Versioning](#3-dvc--data--pipeline-versioning)
4. [Great Expectations — Data Validation](#4-great-expectations--data-validation)
5. [Docker — Containerization](#5-docker--containerization)
6. [FastAPI — Model Serving](#6-fastapi--model-serving)
7. [BentoML — ML-Native Serving](#7-bentoml--ml-native-serving)
8. [Apache Airflow — Pipeline Orchestration](#8-apache-airflow--pipeline-orchestration)
9. [Prefect — Modern Orchestration](#9-prefect--modern-orchestration)
10. [Evidently AI — ML Monitoring](#10-evidently-ai--ml-monitoring)
11. [Prometheus + Grafana — Infrastructure Monitoring](#11-prometheus--grafana--infrastructure-monitoring)
12. [GitHub Actions — CI/CD for ML](#12-github-actions--cicd-for-ml)
13. [Poetry — Dependency Management](#13-poetry--dependency-management)
14. [Pre-commit — Code Quality Automation](#14-pre-commit--code-quality-automation)
15. [Papermill — Notebook Execution in Production](#15-papermill--notebook-execution-in-production)

---

## 1. MLflow — Experiment Tracking

### What It Is

MLflow is the tool you will use on literally your first day of ML work at most companies. It is an open-source platform that tracks experiments (every training run you do), stores model artifacts, manages a model registry, and makes all of this browseable through a clean web UI. Think of it as Git, but for ML experiments instead of code.

> **At work**: Your tech lead will say "before you run any training, set up MLflow tracking." Every experiment you run — every hyperparameter combination you try — gets logged here automatically. When someone asks "which run produced the model in production?", the answer lives in MLflow.

### Install

```bash
pip install mlflow scikit-learn pandas matplotlib seaborn

# Start the UI locally (run this in a terminal, keep it running)
mlflow ui --host 0.0.0.0 --port 5000
# Visit: http://localhost:5000
```

### Full Working Code — Everything You Need

```python
# mlflow_complete.py
# This one file covers 90% of what you'll use MLflow for at work

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import json, os

# ══════════════════════════════════════════════
# STEP 1 — CONNECT TO TRACKING SERVER
# ══════════════════════════════════════════════

# Local development
mlflow.set_tracking_uri("http://localhost:5000")

# At company (remote server — they'll give you this URL)
# mlflow.set_tracking_uri("http://mlflow.company.internal:5000")

# Cloud-managed (Databricks)
# mlflow.set_tracking_uri("databricks")

# ══════════════════════════════════════════════
# STEP 2 — CREATE / USE AN EXPERIMENT
# ══════════════════════════════════════════════

# Each project/feature gets its own experiment
EXPERIMENT_NAME = "customer-churn-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

# ══════════════════════════════════════════════
# STEP 3 — LOAD DATA
# ══════════════════════════════════════════════

# Dummy data — replace with your actual dataset
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'tenure': np.random.randint(0, 120, n),
    'monthly_charges': np.random.uniform(10, 200, n),
    'support_calls': np.random.poisson(2, n),
    'churn': np.random.binomial(1, 0.2, n)
})

X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ══════════════════════════════════════════════
# STEP 4 — LOG A TRAINING RUN
# ══════════════════════════════════════════════

params = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 10,
    "random_state": 42,
}

with mlflow.start_run(run_name="RF-baseline-v1") as run:
    print(f"Run ID: {run.info.run_id}")

    # ── Log hyperparameters ──────────────────────
    mlflow.log_params(params)
    # Extra context params (very useful for teammates)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("features", list(X.columns))
    mlflow.log_param("positive_rate", round(y_train.mean(), 4))

    # ── Tags (use for filtering in UI) ──────────
    mlflow.set_tags({
        "model_type": "random_forest",
        "dataset_version": "v3.2",
        "author": "satyam",
        "environment": "dev",
        "ticket": "ML-142"  # Jira ticket — very helpful in companies
    })

    # ── Train ────────────────────────────────────
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }
    mlflow.log_metrics(metrics)

    # Log step-level metrics (loss per epoch / iteration)
    for i, score in enumerate(
        cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    ):
        mlflow.log_metric("cv_roc_auc", score, step=i)

    # ── Log Artifacts (files) ─────────────────────
    os.makedirs("tmp_artifacts", exist_ok=True)

    # Feature importance plot
    feat_imp = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    feat_imp.plot(kind='barh', ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig("tmp_artifacts/feature_importance.png")
    mlflow.log_artifact("tmp_artifacts/feature_importance.png")
    plt.close()

    # Save feature list as JSON (useful for serving)
    with open("tmp_artifacts/feature_names.json", "w") as f:
        json.dump(list(X.columns), f)
    mlflow.log_artifact("tmp_artifacts/feature_names.json")

    # ── Log the Model ─────────────────────────────
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, y_proba)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
        # Auto-register directly to the registry:
        registered_model_name="churn-model"
    )

    print(f"Metrics: {metrics}")

# ══════════════════════════════════════════════
# STEP 5 — COMPARE RUNS PROGRAMMATICALLY
# ══════════════════════════════════════════════

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# Get best run by ROC AUC
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.roc_auc > 0.8",
    order_by=["metrics.roc_auc DESC"],
    max_results=5
)

print("\nTop 5 runs by ROC AUC:")
for r in runs:
    print(f"  {r.info.run_name}: AUC={r.data.metrics.get('roc_auc'):.4f}")

# ══════════════════════════════════════════════
# STEP 6 — LOAD MODEL FROM MLFLOW
# ══════════════════════════════════════════════

# Load the production model — always use this pattern in serving code
best_run_id = runs[0].info.run_id
model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Or from registry (use in production — decoupled from run_id)
# loaded_model = mlflow.sklearn.load_model("models:/churn-model/Production")

preds = loaded_model.predict_proba(X_test)[:, 1]
print(f"\nLoaded model AUC: {roc_auc_score(y_test, preds):.4f}")

# ══════════════════════════════════════════════
# STEP 7 — MODEL REGISTRY OPERATIONS
# ══════════════════════════════════════════════

client = MlflowClient()

# Transition to staging
versions = client.get_latest_versions("churn-model", stages=["None"])
if versions:
    client.transition_model_version_stage(
        name="churn-model",
        version=versions[0].version,
        stage="Staging"
    )
    print(f"Version {versions[0].version} moved to Staging")

# After testing, promote to production
# client.transition_model_version_stage(
#     name="churn-model", version=2, stage="Production",
#     archive_existing_versions=True
# )
```

### 🔥 Important Tricks

```python
# ── TRICK 1: autolog() saves you 80% of boilerplate ─────────────────
# One line and MLflow automatically logs params, metrics, and model
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=True
)
# Now just train — everything is logged automatically
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ── TRICK 2: Nested runs for hyperparameter search ──────────────────
with mlflow.start_run(run_name="hyperparam-search") as parent_run:
    for n_est in [50, 100, 200]:
        with mlflow.start_run(run_name=f"trial-n{n_est}", nested=True):
            mlflow.log_param("n_estimators", n_est)
            m = RandomForestClassifier(n_estimators=n_est, random_state=42)
            m.fit(X_train, y_train)
            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
            mlflow.log_metric("roc_auc", auc)

# ── TRICK 3: Log custom Python class as MLflow model ────────────────
# Use when your "model" is more than just sklearn — e.g., preprocessing + model
class ChurnPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["scaler_path"], "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame):
        scaled = self.scaler.transform(model_input)
        return self.model.predict_proba(scaled)[:, 1]

# Log it
with mlflow.start_run():
    import pickle
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier().fit(scaler.transform(X_train), y_train)
    pickle.dump(model, open("tmp_artifacts/model.pkl", "wb"))
    pickle.dump(scaler, open("tmp_artifacts/scaler.pkl", "wb"))

    mlflow.pyfunc.log_model(
        artifact_path="churn_predictor",
        python_model=ChurnPredictor(),
        artifacts={
            "model_path": "tmp_artifacts/model.pkl",
            "scaler_path": "tmp_artifacts/scaler.pkl"
        }
    )

# ── TRICK 4: Environment variables for secrets (NEVER hardcode) ─────
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_password"
# Or in .env file (use python-dotenv)

# ── TRICK 5: Delete failed / junk runs to keep UI clean ─────────────
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="status = 'FAILED'"
)
for run in runs:
    client.delete_run(run.info.run_id)
    print(f"Deleted failed run: {run.info.run_id}")

# ── TRICK 6: Tag your runs with git commit hash ─────────────────────
import subprocess
git_hash = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]
).decode().strip()
mlflow.set_tag("git_commit", git_hash)
# Now you can always find the exact code that produced any model
```

> **Common Fresher Mistake**: Running experiments without `mlflow.set_experiment()`. All your runs end up in the "Default" experiment mixed with everyone else's. Always set an experiment name at the top of every training script.

---

## 2. Weights & Biases (W&B)

### What It Is

W&B (Weights & Biases) is a commercial experiment tracking platform that many companies use instead of (or alongside) MLflow. It has a better default UI, richer visualization, and a team collaboration model. If your company uses W&B, you'll interact with it through `wandb` — the Python client. The concepts are identical to MLflow; only the API differs.

### Install

```bash
pip install wandb
wandb login  # Opens browser for auth — do this once
```

### Full Working Code

```python
# wandb_complete.py

import wandb
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── STEP 1: Initialize a Run ─────────────────────────────────────
run = wandb.init(
    project="customer-churn",      # Groups runs into a project
    entity="your-team-name",       # Your W&B team (company's org)
    name="GBM-tuned-v2",           # Human readable run name
    tags=["gradient-boosting", "production-candidate"],
    notes="Tuned learning rate and max_depth after analysis of v1",
    config={                        # Hyperparameters — log all of them
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "random_state": 42,
        "dataset_version": "v4.1",
    }
)

# Access config (good practice — use config object, not raw dicts)
cfg = wandb.config

# Dummy data
np.random.seed(42)
n = 5000
X = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'tenure': np.random.randint(0, 120, n),
    'monthly_charges': np.random.uniform(10, 200, n),
    'support_calls': np.random.poisson(2, n),
})
y = np.random.binomial(1, 0.2, n)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ── STEP 2: Train ────────────────────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=cfg.n_estimators,
    learning_rate=cfg.learning_rate,
    max_depth=cfg.max_depth,
    subsample=cfg.subsample,
    random_state=cfg.random_state
)
model.fit(X_train, y_train)

# ── STEP 3: Log Metrics ──────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.45).astype(int)

wandb.log({
    "roc_auc":  roc_auc_score(y_test, y_proba),
    "f1":       f1_score(y_test, y_pred),
    "n_train":  len(X_train),
    "pos_rate": y_train.mean(),
})

# Log step-by-step metrics (loss curves, useful for neural nets)
for step, score in enumerate(model.train_score_):
    wandb.log({"train_deviance": score}, step=step)

# ── STEP 4: Log Plots ────────────────────────────────────────────
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
plt.tight_layout()
wandb.log({"evaluation_plots": wandb.Image(fig)})
plt.close()

# Log feature importance as a W&B Table (renders as interactive table)
feat_table = wandb.Table(
    columns=["Feature", "Importance"],
    data=[[f, i] for f, i in
          zip(X.columns, model.feature_importances_)]
)
wandb.log({"feature_importance": feat_table})

# ── STEP 5: Save Model as W&B Artifact ──────────────────────────
import pickle, os
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

artifact = wandb.Artifact(
    name="churn-gbm",
    type="model",
    description="GBM churn model v2",
    metadata={"roc_auc": roc_auc_score(y_test, y_proba)}
)
artifact.add_file("models/model.pkl")
run.log_artifact(artifact)

# ── STEP 6: Finish the Run ───────────────────────────────────────
wandb.finish()
```

### 🔥 Important Tricks

```python
# ── TRICK 1: W&B Sweeps — distributed hyperparameter search ─────────
# Define sweep config
sweep_config = {
    "method": "bayes",     # or "grid", "random"
    "metric": {"name": "roc_auc", "goal": "maximize"},
    "parameters": {
        "n_estimators":  {"values": [100, 200, 300, 500]},
        "learning_rate": {"distribution": "log_uniform_values",
                          "min": 0.01, "max": 0.3},
        "max_depth":     {"values": [3, 5, 7, 9]},
    }
}

def train_sweep():
    with wandb.init() as run:
        cfg = run.config
        # Load data...
        model = GradientBoostingClassifier(**dict(cfg))
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        wandb.log({"roc_auc": auc})

# Create sweep and launch agents
sweep_id = wandb.sweep(sweep_config, project="churn-sweeps")
wandb.agent(sweep_id, function=train_sweep, count=50)

# ── TRICK 2: Resume a crashed run ───────────────────────────────────
run = wandb.init(
    project="churn",
    id="your-run-id-here",   # Same ID as the crashed run
    resume="must"            # Will error if run doesn't exist
)

# ── TRICK 3: Offline mode (no internet — common on GPU clusters) ────
import os
os.environ["WANDB_MODE"] = "offline"
# Logs are saved locally. Sync later:
# wandb sync ./wandb/offline-run-*

# ── TRICK 4: Log system metrics automatically ────────────────────────
wandb.init(
    project="my-project",
    settings=wandb.Settings(
        _disable_stats=False,    # Logs CPU, GPU, RAM automatically
        console="auto"           # Captures stdout/stderr
    )
)

# ── TRICK 5: Compare runs programmatically ───────────────────────────
api = wandb.Api()
runs = api.runs(
    "your-team/customer-churn",
    filters={"tags": {"$in": ["production-candidate"]}}
)
for run in runs:
    print(f"{run.name}: AUC={run.summary.get('roc_auc', 'N/A'):.4f}")
```

---

## 3. DVC — Data & Pipeline Versioning

### What It Is

DVC (Data Version Control) solves a problem Git cannot: versioning large datasets, feature files, and model binaries. At work, you'll use DVC to ensure every training run can be reproduced exactly — same code (Git) + same data (DVC) = identical results. Many companies also use DVC pipelines to define the ML workflow as code.

### Install

```bash
pip install dvc dvc-s3  # or dvc-gs for GCS, dvc-azure for Azure

git init my-project && cd my-project
dvc init
git add . && git commit -m "Initialize DVC"
```

### Full Working Code

```bash
# ── 1. Configure Remote Storage ──────────────────────────────────────
# S3
dvc remote add -d myremote s3://my-company-bucket/dvc-store
# GCS
dvc remote add -d myremote gs://my-company-bucket/dvc-store
# Azure
dvc remote add -d myremote azure://my-container/dvc-store
# Local (for learning)
dvc remote add -d myremote /tmp/dvc-remote-storage

git add .dvc/config
git commit -m "Configure DVC remote"

# ── 2. Track a Dataset ───────────────────────────────────────────────
dvc add data/raw/customers.csv
# Creates: data/raw/customers.csv.dvc (tiny metadata file)
# Adds: data/raw/customers.csv to .gitignore

git add data/raw/customers.csv.dvc data/raw/.gitignore
git commit -m "Add customers dataset v1"
dvc push  # Uploads actual data to S3/GCS

# ── 3. Pull data on new machine ──────────────────────────────────────
git clone https://github.com/your-org/your-repo
cd your-repo
dvc pull  # Downloads data from remote

# ── 4. Update data (new version) ────────────────────────────────────
# After updating customers.csv:
dvc add data/raw/customers.csv   # Recomputes hash
git add data/raw/customers.csv.dvc
git commit -m "Update customers dataset v2 — added new columns"
dvc push

# Go back to v1 data at any time:
git checkout HEAD~1 -- data/raw/customers.csv.dvc
dvc checkout  # Fetches the v1 data from remote
```

### DVC Pipeline (The Most Valuable Feature)

```yaml
# dvc.yaml — define your entire ML pipeline as code
stages:

  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/customers.csv      # If this changes, stage re-runs
    params:
      - params.yaml:                 # If params change, stage re-runs
          - prepare.test_size
          - prepare.random_seed
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv    # Depends on previous stage output
    params:
      - params.yaml:
          - train.n_estimators
          - train.max_depth
          - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false              # Track in git, not DVC cache
```

```yaml
# params.yaml
prepare:
  test_size: 0.2
  random_seed: 42

train:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
```

```python
# src/train.py — reads params, writes metrics
import yaml, json, pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
X_train, y_train = train_df.drop("churn", axis=1), train_df["churn"]
X_test,  y_test  = test_df.drop("churn",  axis=1), test_df["churn"]

model = GradientBoostingClassifier(**params, random_state=42)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "f1":      round(f1_score(y_test, (y_proba >= 0.45).astype(int)), 4),
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics: {metrics}")

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

```bash
# Run pipeline (only re-runs changed stages)
dvc repro

# ── See metrics ──────────────────────────────────────────────────────
dvc metrics show
# Path                  roc_auc    f1
# metrics/scores.json   0.8923     0.6341

# Compare current vs last commit
dvc metrics diff HEAD~1

# ── Run experiment with different params ─────────────────────────────
dvc exp run --set-param train.n_estimators=300 --set-param train.max_depth=8
# or shorthand:
dvc exp run -S train.n_estimators=300

# See all experiments side by side
dvc exp show --md   # Markdown table (great for reports)

# Apply best experiment to your workspace
dvc exp apply exp-abc123
```

### 🔥 Important Tricks

```bash
# ── TRICK 1: .dvcignore for large auto-generated files ───────────────
echo "data/processed/__pycache__" >> .dvcignore
echo "*.pyc" >> .dvcignore

# ── TRICK 2: dvc pull --run-cache fetches cached pipeline results ────
dvc pull --run-cache
# If someone else ran the pipeline with these exact inputs,
# you get their outputs instantly — no re-training needed!

# ── TRICK 3: Use dvc.lock to see exactly what ran ───────────────────
cat dvc.lock
# Shows the MD5 hash of every input and output of every stage.
# This is your reproducibility guarantee.

# ── TRICK 4: Tag experiments before sharing ─────────────────────────
git tag -a "experiment-v3-best" -m "Best model before Q2 release"
dvc push  # Push data for this tag

# Teammate can reproduce this exact state:
git checkout experiment-v3-best
dvc checkout
dvc repro

# ── TRICK 5: Run multiple parameter combos in parallel ───────────────
dvc exp run \
  --set-param train.n_estimators=100,200,300 \
  --set-param train.max_depth=4,6,8 \
  --jobs 4   # Run 4 experiments in parallel
```

---

## 4. Great Expectations — Data Validation

### What It Is

Great Expectations (GX) is the industry-standard tool for validating data quality. At work, it runs as a step in your pipeline before training starts — if data doesn't meet expectations (wrong schema, missing columns, out-of-range values), the pipeline fails loudly instead of training a silently broken model.

### Install

```bash
pip install great_expectations
great_expectations init  # Creates ge_dir/ with configuration
```

### Full Working Code

```python
# data_validation.py — complete GX v3 API usage

import great_expectations as gx
import pandas as pd
import numpy as np

# ── OPTION A: Quick validation (no project setup needed) ────────────
# Great for scripts and one-off checks

context = gx.get_context()

# Point at your data
validator = context.sources.pandas_default.read_csv(
    "data/raw/customers.csv"
)

# Define what you EXPECT the data to look like
validator.expect_table_row_count_to_be_between(min_value=1000)
validator.expect_column_to_exist("customer_id")
validator.expect_column_to_exist("churn")
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_not_be_null("churn")
validator.expect_column_values_to_be_unique("customer_id")
validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)
validator.expect_column_values_to_be_between("monthly_charges", min_value=0)
validator.expect_column_values_to_be_in_set("churn", value_set=[0, 1])
validator.expect_column_values_to_be_in_set(
    "plan_type",
    value_set=["free", "basic", "premium", "enterprise"]
)
validator.expect_column_proportion_of_unique_values_to_be_between(
    "customer_id", min_value=0.99  # Allow 1% duplicates max
)

# Run validation
result = validator.validate()
print(f"Validation passed: {result.success}")
if not result.success:
    # Print failed expectations
    for r in result.results:
        if not r.success:
            print(f"  FAIL: {r.expectation_config.expectation_type} "
                  f"on column '{r.expectation_config.kwargs.get('column', 'table')}'")
    raise ValueError("Data validation failed — stopping pipeline!")


# ── OPTION B: Full project setup (use this at work) ─────────────────

def run_production_validation(data_path: str, suite_name: str) -> bool:
    """
    Full GX validation with checkpoint.
    Returns True if passed, raises on failure.
    """
    context = gx.get_context()

    # Add datasource
    try:
        datasource = context.get_datasource("my_datasource")
    except Exception:
        datasource = context.sources.add_pandas_filesystem(
            name="my_datasource",
            base_directory="data/"
        )

    # Add asset
    asset = datasource.add_csv_asset(
        name="customer_data",
        batching_regex=r"(?P<name>.+)\.csv"
    )
    batch_request = asset.build_batch_request(
        options={"name": data_path.replace("data/", "").replace(".csv", "")}
    )

    # Get or create suite
    try:
        suite = context.get_expectation_suite(suite_name)
    except Exception:
        suite = context.add_expectation_suite(suite_name)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Add expectations
    validator.expect_table_row_count_to_be_between(min_value=100)
    validator.expect_column_to_exist("customer_id")
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_between("age", 18, 120)

    validator.save_expectation_suite()

    # Create and run checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name=f"{suite_name}_checkpoint",
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": suite_name
        }]
    )

    result = checkpoint.run()

    # Build and open data docs (HTML report — send to stakeholders)
    context.build_data_docs()

    return result.success
```

### 🔥 Important Tricks

```python
# ── TRICK 1: Custom expectation for ML-specific checks ───────────────
# Check that class imbalance is within acceptable range
@validator.expect_column_proportion_of_unique_values_to_be_between(
    "churn",
    min_value=0.05,   # At least 5% positive class
    max_value=0.5     # At most 50% positive class
)

# ── TRICK 2: Profiling — auto-generate expectations from data ────────
from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler
import great_expectations as gx

df = pd.read_csv("data/raw/customers.csv")
ge_df = gx.from_pandas(df)

# Auto-profile generates expectations from your data's statistics
suite, validation_result = BasicDatasetProfiler.profile(ge_df)
print(f"Auto-generated {len(suite.expectations)} expectations")

# ── TRICK 3: Use GX in Airflow DAG ───────────────────────────────────
from airflow.operators.python import PythonOperator

def validate_task(**context):
    result = run_production_validation("data/raw/customers.csv", "prod_suite")
    if not result:
        raise ValueError("Data validation failed")

validate_op = PythonOperator(
    task_id="validate_data",
    python_callable=validate_task,
    dag=dag
)

# ── TRICK 4: Soft vs hard assertions ────────────────────────────────
# Hard: pipeline stops if failed (use for critical checks)
validator.expect_column_values_to_not_be_null(
    "customer_id",
    result_format={"result_format": "COMPLETE"},
    catch_exceptions=False  # Hard fail
)

# Soft: log warning but continue (use for monitoring/alerts)
result = validator.expect_column_mean_to_be_between(
    "monthly_charges", min_value=20, max_value=150
)
if not result.success:
    import logging
    logging.warning(f"Monthly charges mean is unusual: "
                    f"{result.result.get('observed_value')}")
    # Continue pipeline but alert team
```

---

## 5. Docker — Containerization

### What It Is

Docker packages your ML model, code, and all dependencies into a single portable container. Without Docker, "it works on my machine" is a constant problem in ML deployment. With Docker, the exact same container runs on your laptop, CI server, and production Kubernetes cluster. At work, every model serving endpoint runs in a Docker container.

### Full Working Code

```dockerfile
# Dockerfile — production-ready ML model container

# Use specific version — never use "latest" in production
FROM python:3.11.7-slim

# Metadata
LABEL maintainer="satyam@company.com"
LABEL version="2.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies (do this before Python deps for cache efficiency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (security best practice — companies require this)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy and install requirements BEFORE copying code
# This way, code changes don't invalidate the pip layer (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY models/ models/
COPY config/ config/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV MODEL_PATH=/app/models/model.pkl
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check — Kubernetes uses this to know when your pod is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

CMD ["uvicorn", "src.app:app",
     "--host", "0.0.0.0",
     "--port", "8080",
     "--workers", "2"]
```

```bash
# ── Build ─────────────────────────────────────────────────────────────
docker build -t churn-model:v2.1.0 .

# Build with build args (useful for versioning)
docker build \
  --build-arg MODEL_VERSION=2.1.0 \
  --build-arg GIT_HASH=$(git rev-parse --short HEAD) \
  -t churn-model:v2.1.0 .

# ── Run locally for testing ───────────────────────────────────────────
docker run -d \
  --name churn-test \
  -p 8080:8080 \
  -e MODEL_PATH=/app/models/model.pkl \
  -v $(pwd)/models:/app/models:ro \   # Mount model volume (read-only)
  churn-model:v2.1.0

# Test it
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 32, "tenure": 18, "monthly_charges": 79.5}'

# View logs
docker logs churn-test -f

# Stop and remove
docker stop churn-test && docker rm churn-test

# ── Push to Registry ─────────────────────────────────────────────────
# GCP Artifact Registry
docker tag churn-model:v2.1.0 \
  us-central1-docker.pkg.dev/my-project/ml-models/churn-model:v2.1.0
docker push \
  us-central1-docker.pkg.dev/my-project/ml-models/churn-model:v2.1.0

# AWS ECR
aws ecr get-login-password | docker login --username AWS \
  --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/churn-model:v2.1.0
```

### docker-compose for Local MLOps Stack

```yaml
# docker-compose.yml — run full local MLOps stack
version: '3.9'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:password@db/mlflow
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    depends_on:
      - db
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:password@db/mlflow
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0

  model-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    volumes:
      - ./models:/app/models:ro
    depends_on:
      - mlflow

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mlflow
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  mlflow-artifacts:
  pgdata:
```

```bash
# Start full stack
docker-compose up -d

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build model-server
```

### 🔥 Important Tricks

```dockerfile
# ── TRICK 1: Multi-stage build — smaller final image ─────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
# Install to user dir so we can copy cleanly
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
# Copy only installed packages, not the build tools
COPY --from=builder /root/.local /root/.local
COPY src/ src/
COPY models/ models/
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
# Result: ~300MB instead of ~1.2GB

# ── TRICK 2: .dockerignore — speed up builds, reduce image size ──────
# Create .dockerignore:
# .git/
# .dvc/
# __pycache__/
# *.pyc
# .env
# tests/
# notebooks/
# data/raw/          # Never include training data in image!
# *.ipynb
```

```bash
# ── TRICK 3: Inspect what's in an image ──────────────────────────────
docker image inspect churn-model:v2.1.0
docker history churn-model:v2.1.0       # See layer sizes — find bloat

# ── TRICK 4: Debug a running container ───────────────────────────────
docker exec -it churn-test /bin/bash    # Shell into running container
docker exec -it churn-test python -c "import sklearn; print(sklearn.__version__)"

# ── TRICK 5: Check resource usage ────────────────────────────────────
docker stats churn-test   # Live CPU/memory — use to right-size k8s resources
```

---

## 6. FastAPI — Model Serving

### What It Is

FastAPI is the standard framework for wrapping ML models in REST APIs. It's fast (built on Starlette + Pydantic), auto-generates Swagger documentation, and uses Python type hints for input validation. Every time you need a model to respond to real-time HTTP requests, you'll write a FastAPI app.

### Full Working Code

```python
# src/app.py — production FastAPI model server

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import mlflow.sklearn
import pandas as pd
import numpy as np
import time, logging, os, pickle
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────────
model = None
model_version = "unknown"

# ── Lifespan (modern FastAPI startup/shutdown) ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, model_version
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_version = os.getenv("MODEL_VERSION", "v1")
        logger.info(f"Model loaded from {model_path} (version {model_version})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # Shutdown — cleanup here

app = FastAPI(
    title="Churn Prediction API",
    version="2.1.0",
    lifespan=lifespan
)

# ── Prometheus metrics ────────────────────────────────────────────────
REQUESTS = Counter("http_requests_total", "Total HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY  = Histogram("request_latency_seconds", "Request latency",
                     ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Total predictions",
                      ["outcome", "model_version"])

# ── Request/Response schemas (Pydantic) ───────────────────────────────
class CustomerInput(BaseModel):
    customer_id: str = Field(..., description="Unique customer ID")
    age: int = Field(..., ge=18, le=120)
    tenure: int = Field(..., ge=0, description="Months as customer")
    monthly_charges: float = Field(..., gt=0)
    support_calls: int = Field(..., ge=0)
    plan_type: str = Field(..., description="free|basic|premium|enterprise")

    @validator("plan_type")
    def validate_plan(cls, v):
        valid = {"free", "basic", "premium", "enterprise"}
        if v not in valid:
            raise ValueError(f"plan_type must be one of {valid}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "C001",
                "age": 32, "tenure": 18,
                "monthly_charges": 79.5,
                "support_calls": 1,
                "plan_type": "premium"
            }
        }

class PredictionOutput(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    risk_level: str   # LOW | MEDIUM | HIGH
    model_version: str

class BatchRequest(BaseModel):
    customers: List[CustomerInput]

class BatchResponse(BaseModel):
    predictions: List[PredictionOutput]
    total_processed: int
    inference_ms: float

# ── Helper ────────────────────────────────────────────────────────────
PLAN_MAP = {"free": 0, "basic": 1, "premium": 2, "enterprise": 3}

def to_features(c: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "age": c.age,
        "tenure": c.tenure,
        "monthly_charges": c.monthly_charges,
        "support_calls": c.support_calls,
        "plan_encoded": PLAN_MAP[c.plan_type]
    }])

def risk_level(p: float) -> str:
    return "HIGH" if p >= 0.6 else "MEDIUM" if p >= 0.3 else "LOW"

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Kubernetes liveness/readiness probe endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }

@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictionOutput)
async def predict_single(customer: CustomerInput):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()
    try:
        features = to_features(customer)
        proba = float(model.predict_proba(features)[0, 1])
        latency = (time.perf_counter() - start) * 1000

        outcome = "churn" if proba >= 0.45 else "retain"
        PREDICTIONS.labels(outcome=outcome, model_version=model_version).inc()
        LATENCY.labels(endpoint="/predict").observe(latency / 1000)

        return PredictionOutput(
            customer_id=customer.customer_id,
            churn_probability=round(proba, 4),
            churn_predicted=proba >= 0.45,
            risk_level=risk_level(proba),
            model_version=model_version
        )
    except Exception as e:
        REQUESTS.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()
    features_df = pd.concat(
        [to_features(c) for c in request.customers],
        ignore_index=True
    )
    probas = model.predict_proba(features_df)[:, 1]
    latency = (time.perf_counter() - start) * 1000

    predictions = [
        PredictionOutput(
            customer_id=c.customer_id,
            churn_probability=round(float(p), 4),
            churn_predicted=float(p) >= 0.45,
            risk_level=risk_level(float(p)),
            model_version=model_version
        )
        for c, p in zip(request.customers, probas)
    ]

    return BatchResponse(
        predictions=predictions,
        total_processed=len(predictions),
        inference_ms=round(latency, 2)
    )
```

### 🔥 Important Tricks

```python
# ── TRICK 1: Middleware for request logging ──────────────────────────
from fastapi import Request
import uuid

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    logger.info(
        f"id={request_id} method={request.method} "
        f"path={request.url.path} status={response.status_code} "
        f"latency={latency:.1f}ms"
    )
    response.headers["X-Request-ID"] = request_id
    return response

# ── TRICK 2: Background tasks (for async logging, no latency hit) ────
from fastapi import BackgroundTasks

def log_prediction_async(customer_id: str, prediction: float):
    # Write to DB, Pub/Sub, S3, etc. — happens after response is sent
    pass

@app.post("/predict")
async def predict(customer: CustomerInput, background_tasks: BackgroundTasks):
    proba = float(model.predict_proba(to_features(customer))[0, 1])
    background_tasks.add_task(log_prediction_async, customer.customer_id, proba)
    return {"churn_probability": proba}

# ── TRICK 3: Dependency injection for shared resources ───────────────
from fastapi import Depends

def get_model():
    if model is None:
        raise HTTPException(503, "Model not available")
    return model

@app.post("/predict")
async def predict(customer: CustomerInput, m=Depends(get_model)):
    return {"prob": float(m.predict_proba(to_features(customer))[0, 1])}

# ── TRICK 4: Custom exception handler ────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)}
    )

# ── TRICK 5: Test your FastAPI app (essential for CI) ────────────────
# tests/test_app.py
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "customer_id": "C001", "age": 32, "tenure": 18,
        "monthly_charges": 79.5, "support_calls": 1, "plan_type": "premium"
    })
    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1

def test_invalid_plan_type():
    response = client.post("/predict", json={
        "customer_id": "C001", "age": 32, "tenure": 18,
        "monthly_charges": 79.5, "support_calls": 1,
        "plan_type": "invalid_plan"  # Should be rejected
    })
    assert response.status_code == 422
```

---

## 7. BentoML — ML-Native Serving

### What It Is

BentoML is purpose-built for ML serving. Unlike FastAPI (general web framework), BentoML understands ML natively — it handles model storage, adaptive batching (combining concurrent requests into single batch calls for 10x throughput), and containerization automatically. Many companies use BentoML to serve high-throughput models.

### Full Working Code

```python
# bento_service.py

import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import pickle

# ── STEP 1: Save your model to BentoML store ────────────────────────
with open("models/model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# Save to BentoML's local model store
bento_model = bentoml.sklearn.save_model(
    name="churn_model",         # Model name
    model=sklearn_model,
    signatures={
        # Tell BentoML these methods can be batched
        "predict_proba": {"batchable": True, "batch_dim": 0},
    },
    metadata={
        "accuracy": 0.89,
        "trained_on": "2024-01-15",
        "features": ["age", "tenure", "monthly_charges", "support_calls"]
    }
)
print(f"Saved: {bento_model.tag}")  # e.g., churn_model:abc123xyz

# ── STEP 2: Create a service ─────────────────────────────────────────
runner = bentoml.sklearn.get("churn_model:latest").to_runner()

svc = bentoml.Service(name="churn_prediction", runners=[runner])

class PredictInput(BaseModel):
    customer_id: str
    age: int
    tenure: int
    monthly_charges: float
    support_calls: int

class PredictOutput(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool

@svc.api(input=JSON(pydantic_model=PredictInput),
         output=JSON(pydantic_model=PredictOutput))
async def predict(inp: PredictInput) -> PredictOutput:
    features = np.array([[
        inp.age, inp.tenure,
        inp.monthly_charges, inp.support_calls
    ]])
    proba = await runner.predict_proba.async_run(features)
    churn_proba = float(proba[0, 1])
    return PredictOutput(
        customer_id=inp.customer_id,
        churn_probability=round(churn_proba, 4),
        churn_predicted=churn_proba >= 0.45
    )
```

```bash
# Serve locally
bentoml serve bento_service:svc --reload

# Build a Bento (portable package with all deps)
bentoml build

# Containerize
bentoml containerize churn_prediction:latest

# Run container
docker run -p 3000:3000 churn_prediction:latest serve

# List saved models
bentoml models list

# List bentos
bentoml list
```

---

## 8. Apache Airflow — Pipeline Orchestration

### What It Is

Airflow is the most used tool for scheduling and orchestrating ML pipelines in production. You write Python code that defines a DAG (Directed Acyclic Graph) of tasks. Airflow schedules runs, retries failed tasks, sends alerts, and provides a UI to monitor everything. At work, your nightly training pipeline, daily feature computation, and weekly retraining jobs almost certainly run on Airflow.

### Install (Local Dev)

```bash
pip install "apache-airflow[postgres,redis,celery]==2.8.0"

# Or Docker Compose (easiest for local dev)
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
docker-compose up -d
# Visit: http://localhost:8080  (user: airflow, pass: airflow)
```

### Full Working ML DAG

```python
# dags/churn_training_pipeline.py
# Put this file in your airflow/dags/ folder

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging

logger = logging.getLogger(__name__)

# ── Default args applied to all tasks ─────────────────────────────────
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

# ── Python callables (the actual work) ────────────────────────────────

def ingest_data(**context):
    """Pull latest data from data warehouse."""
    logger.info("Starting data ingestion...")
    # Simulate ingestion
    import pandas as pd, numpy as np
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, 1000),
        "tenure": np.random.randint(0, 120, 1000),
        "monthly_charges": np.random.uniform(10, 200, 1000),
        "support_calls": np.random.poisson(2, 1000),
        "churn": np.random.binomial(1, 0.2, 1000)
    })
    df.to_csv("/tmp/customers.csv", index=False)

    # Pass data to next task via XCom
    context["ti"].xcom_push(key="data_path", value="/tmp/customers.csv")
    context["ti"].xcom_push(key="row_count", value=len(df))
    logger.info(f"Ingested {len(df)} rows")

def validate_data(**context):
    """Run Great Expectations validation."""
    import pandas as pd
    data_path = context["ti"].xcom_pull(key="data_path")
    df = pd.read_csv(data_path)

    assert len(df) > 100, "Too few rows!"
    assert "churn" in df.columns, "Missing churn column!"
    assert df["churn"].isnull().sum() == 0, "Nulls in churn column!"
    logger.info("Data validation passed ✓")

def train_model(**context):
    """Train the model and log to MLflow."""
    import mlflow, pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    data_path = context["ti"].xcom_pull(key="data_path")
    df = pd.read_csv(data_path)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("airflow-production")

    with mlflow.start_run(run_name=f"airflow-run-{context['ds']}") as run:
        model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        context["ti"].xcom_push(key="roc_auc", value=auc)
        context["ti"].xcom_push(key="run_id", value=run.info.run_id)
    logger.info(f"Training complete. ROC AUC: {auc:.4f}")

def check_model_quality(**context):
    """Branch: deploy if good, reject if not."""
    roc_auc = context["ti"].xcom_pull(key="roc_auc")
    THRESHOLD = 0.80
    logger.info(f"ROC AUC: {roc_auc:.4f}, Threshold: {THRESHOLD}")
    if roc_auc >= THRESHOLD:
        return "deploy_model"
    else:
        return "reject_model"

def deploy_model(**context):
    run_id = context["ti"].xcom_pull(key="run_id")
    logger.info(f"Deploying model from run {run_id}")
    # Add actual deployment logic here

def reject_model(**context):
    roc_auc = context["ti"].xcom_pull(key="roc_auc")
    logger.warning(f"Model rejected: ROC AUC {roc_auc:.4f} below threshold")

def send_notification(**context):
    roc_auc = context["ti"].xcom_pull(task_ids="train_model", key="roc_auc")
    logger.info(f"Pipeline complete. Final AUC: {roc_auc}")

# ── Define the DAG ────────────────────────────────────────────────────
with DAG(
    dag_id="churn_training_pipeline",
    default_args=default_args,
    description="Weekly churn model training pipeline",
    schedule_interval="0 2 * * 1",  # Every Monday at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "production", "churn"],
    max_active_runs=1,
    doc_md="""
    ## Churn Training Pipeline
    Trains a GBM churn model weekly.
    - Data from BigQuery
    - Logs to MLflow
    - Deploys to Cloud Run if AUC >= 0.80
    """
) as dag:

    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )
    quality_check = BranchPythonOperator(
        task_id="check_model_quality",
        python_callable=check_model_quality,
    )
    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )
    reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )
    notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_notification,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # Runs after either deploy or reject
    )
    end = EmptyOperator(task_id="end")

    # Wire up the DAG
    start >> ingest >> validate >> train >> quality_check
    quality_check >> [deploy, reject]
    [deploy, reject] >> notify >> end
```

### 🔥 Important Tricks

```python
# ── TRICK 1: XCom best practice — don't pass large data! ─────────────
# BAD: passing a 100MB DataFrame through XCom (will crash Airflow)
context["ti"].xcom_push(key="data", value=huge_dataframe)

# GOOD: pass a file path or GCS/S3 URI
context["ti"].xcom_push(key="data_path", value="gs://bucket/data.parquet")

# ── TRICK 2: TaskGroup for visual organization ────────────────────────
from airflow.utils.task_group import TaskGroup

with DAG("my_pipeline", ...) as dag:
    with TaskGroup("data_preparation") as data_prep:
        ingest = PythonOperator(task_id="ingest", ...)
        validate = PythonOperator(task_id="validate", ...)
        ingest >> validate

    with TaskGroup("model_training") as training:
        train = PythonOperator(task_id="train", ...)
        evaluate = PythonOperator(task_id="evaluate", ...)
        train >> evaluate

    data_prep >> training

# ── TRICK 3: Trigger DAG from external system ─────────────────────────
# Via REST API (trigger from GitHub Actions, monitoring alert, etc.)
import requests
response = requests.post(
    "http://airflow.company.internal:8080/api/v1/dags/churn_training/dagRuns",
    json={"conf": {"triggered_by": "drift_alert", "drift_rate": 0.45}},
    auth=("airflow", "airflow_password")
)

# ── TRICK 4: Dynamic tasks (run N parallel training jobs) ─────────────
from airflow.decorators import task, dag

@dag(schedule_interval="@weekly", start_date=datetime(2024, 1, 1))
def dynamic_training():
    @task
    def get_model_configs():
        return [
            {"model": "random_forest", "n_est": 100},
            {"model": "gradient_boosting", "n_est": 200},
            {"model": "xgboost", "n_est": 150},
        ]

    @task
    def train_one(config: dict):
        print(f"Training {config['model']} with n_est={config['n_est']}")
        return {"model": config["model"], "auc": 0.85}

    configs = get_model_configs()
    results = train_one.expand(config=configs)  # Runs in parallel!

dag_instance = dynamic_training()
```

---

## 9. Prefect — Modern Orchestration

### What It Is

Prefect is a modern alternative to Airflow with a significantly better developer experience. No XML config, no complex setup, pure Python. You decorate your functions and Prefect handles scheduling, retries, and monitoring. Many newer companies choose Prefect over Airflow for new ML projects.

### Install

```bash
pip install prefect
prefect server start  # Start local UI at http://localhost:4200
```

### Full Working Code

```python
# prefect_pipeline.py

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ── Tasks (decorated functions with retry logic) ──────────────────────

@task(
    name="Ingest Customer Data",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,      # Cache based on inputs
    cache_expiration=timedelta(hours=1) # Cache valid for 1 hour
)
def ingest_data(source_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Ingesting data from {source_path}")
    # Simulate
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, 2000),
        "tenure": np.random.randint(0, 120, 2000),
        "monthly_charges": np.random.uniform(10, 200, 2000),
        "support_calls": np.random.poisson(2, 2000),
        "churn": np.random.binomial(1, 0.2, 2000)
    })
    logger.info(f"Loaded {len(df)} rows")
    return df

@task(name="Validate Data")
def validate_data(df: pd.DataFrame) -> bool:
    logger = get_run_logger()
    assert len(df) > 100, "Too few rows"
    assert "churn" in df.columns, "Missing churn"
    assert df.isnull().sum().sum() == 0, "Has nulls"
    logger.info("Validation passed ✓")
    return True

@task(name="Train Model", retries=1)
def train_model(df: pd.DataFrame, n_estimators: int = 200) -> dict:
    logger = get_run_logger()
    X = df.drop("churn", axis=1)
    y = df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    logger.info(f"Training complete. AUC: {auc:.4f}")
    return {"auc": auc, "n_estimators": n_estimators}

@task(name="Evaluate Quality Gate")
def quality_gate(metrics: dict, threshold: float = 0.80) -> bool:
    passed = metrics["auc"] >= threshold
    get_run_logger().info(
        f"Quality gate {'PASSED' if passed else 'FAILED'}: "
        f"AUC {metrics['auc']:.4f} (threshold {threshold})"
    )
    return passed

# ── Flow (the DAG) ────────────────────────────────────────────────────

@flow(
    name="Churn Training Pipeline",
    description="Weekly churn model training and deployment",
    log_prints=True
)
def churn_pipeline(
    source_path: str = "data/customers.csv",
    n_estimators: int = 200,
    quality_threshold: float = 0.80
):
    # Step 1: Ingest
    df = ingest_data(source_path)

    # Step 2: Validate
    valid = validate_data(df)

    # Step 3: Train (only if valid)
    if valid:
        metrics = train_model(df, n_estimators=n_estimators)

        # Step 4: Quality gate
        approved = quality_gate(metrics, threshold=quality_threshold)

        if approved:
            print(f"✅ Model approved! AUC: {metrics['auc']:.4f}")
        else:
            print(f"❌ Model rejected. AUC: {metrics['auc']:.4f}")
    else:
        raise ValueError("Data validation failed")

if __name__ == "__main__":
    # Run locally
    churn_pipeline()

    # Schedule it
    churn_pipeline.serve(
        name="weekly-churn-training",
        cron="0 2 * * 1",    # Every Monday 2 AM
        parameters={"n_estimators": 300, "quality_threshold": 0.82}
    )
```

---

## 10. Evidently AI — ML Monitoring

### What It Is

Evidently is the go-to library for ML-specific monitoring. It generates statistical reports comparing your training (reference) data against current production data, detects feature drift, target drift, and model performance changes. At work, you'll run Evidently as part of your monitoring job to generate reports and trigger retraining alerts.

### Install

```bash
pip install evidently
```

### Full Working Code

```python
# monitoring/drift_monitor.py

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metric_preset import (
    DataDriftPreset, DataQualityPreset,
    TargetDriftPreset, ClassificationPreset
)
from evidently.metrics import (
    DatasetDriftMetric, DataDriftTable,
    ColumnDriftMetric, ColumnDistributionMetric,
    ClassificationQualityMetric
)
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
    TestAccuracyScore
)
from evidently.test_preset import DataStabilityTestPreset

# ── Load reference and current data ──────────────────────────────────
np.random.seed(42)
reference_df = pd.DataFrame({
    "age": np.random.normal(40, 12, 1000).clip(18, 80).astype(int),
    "tenure": np.random.randint(0, 120, 1000),
    "monthly_charges": np.random.normal(80, 30, 1000).clip(10, 200),
    "support_calls": np.random.poisson(1.5, 1000),
    "prediction": np.random.uniform(0, 1, 1000),   # Model output
    "target": np.random.binomial(1, 0.2, 1000),    # Actual labels
})

# Current data (simulate drift — age distribution shifted)
current_df = pd.DataFrame({
    "age": np.random.normal(50, 15, 1000).clip(18, 80).astype(int),  # SHIFTED
    "tenure": np.random.randint(0, 120, 1000),
    "monthly_charges": np.random.normal(90, 35, 1000).clip(10, 200),  # SHIFTED
    "support_calls": np.random.poisson(1.5, 1000),
    "prediction": np.random.uniform(0, 1, 1000),
    "target": np.random.binomial(1, 0.25, 1000),   # Label shift too
})

# ── REPORT 1: Data Drift ─────────────────────────────────────────────
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(num_stattest="ks", cat_stattest="chi2"),
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="monthly_charges"),
    ColumnDistributionMetric(column_name="support_calls"),
])
drift_report.run(reference_data=reference_df,
                  current_data=current_df)
drift_report.save_html("reports/drift_report.html")

# Get results programmatically
results = drift_report.as_dict()
drift_share = results["metrics"][0]["result"]["drift_share"]
print(f"Share of drifted features: {drift_share:.1%}")
if drift_share > 0.3:
    print("🚨 ALERT: >30% of features drifted — consider retraining!")

# ── REPORT 2: Model Performance ──────────────────────────────────────
perf_report = Report(metrics=[
    ClassificationPreset(probas_threshold=0.45),
])
perf_report.run(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=None
)
perf_report.save_html("reports/performance_report.html")

# ── TEST SUITE (pass/fail — use in CI/CD) ────────────────────────────
test_suite = TestSuite(tests=[
    DataStabilityTestPreset(),
    TestNumberOfDriftedColumns(lt=3),         # Fail if >=3 features drift
    TestShareOfDriftedColumns(lt=0.5),        # Fail if >=50% features drift
    TestColumnDrift(column_name="age",
                    stattest="ks",
                    stattest_threshold=0.05),
])
test_suite.run(reference_data=reference_df, current_data=current_df)
test_suite.save_html("reports/test_suite.html")

passed = test_suite.as_dict()["summary"]["all_passed"]
print(f"Tests passed: {passed}")
if not passed:
    print("❌ Monitoring tests FAILED — retraining triggered")
    # trigger_retraining()
```

### 🔥 Important Tricks

```python
# ── TRICK 1: Automated monitoring in production ──────────────────────
def run_weekly_monitoring():
    """Run every week as a cron job or Airflow task."""
    ref = pd.read_parquet("data/features/train_features.parquet")
    cur = pd.read_parquet(
        f"data/production/week_{get_current_week()}.parquet"
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    results = report.as_dict()
    drift_share = results["metrics"][0]["result"]["drift_share"]

    # Log to MLflow for tracking over time
    with mlflow.start_run(experiment_id="monitoring"):
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_artifact("reports/drift_report.html")

    # Alert if needed
    if drift_share > 0.3:
        send_slack_alert(f"⚠️ High drift detected: {drift_share:.1%}")

# ── TRICK 2: Monitoring dashboard with Evidently + Streamlit ─────────
# streamlit_monitor.py
import streamlit as st
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

st.title("ML Model Monitoring Dashboard")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

# Embed Evidently HTML report directly in Streamlit
import streamlit.components.v1 as components
report.save_html("/tmp/report.html")
with open("/tmp/report.html") as f:
    components.html(f.read(), height=1000, scrolling=True)
```

---

## 11. Prometheus + Grafana — Infrastructure Monitoring

### What It Is

Prometheus scrapes metrics from your running services (request count, latency, error rate, memory usage) and stores them as time-series data. Grafana reads from Prometheus and renders beautiful dashboards. At work, every deployed model endpoint has Prometheus + Grafana monitoring — it's how the team knows if the service is healthy right now.

### FastAPI + Prometheus Integration

```python
# Already shown in FastAPI section — the key part:
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

REQUESTS   = Counter("http_requests_total",
                      "Total requests", ["method", "endpoint", "status"])
LATENCY    = Histogram("request_latency_seconds",
                        "Latency", ["endpoint"],
                        buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5])
MODEL_PRED = Histogram("model_prediction_value",
                        "Distribution of prediction values",
                        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ERRORS     = Counter("prediction_errors_total", "Prediction errors")
MODEL_VER  = Gauge("model_version_info", "Model version",
                    ["version"])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Prometheus Config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'churn-model'
    static_configs:
      - targets: ['churn-model-service:8080']
    metrics_path: /metrics

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana Dashboard JSON (Key Panels)

```json
{
  "panels": [
    {
      "title": "Requests Per Second",
      "type": "graph",
      "targets": [{
        "expr": "rate(http_requests_total[5m])",
        "legendFormat": "{{endpoint}} {{status}}"
      }]
    },
    {
      "title": "P99 Latency",
      "type": "graph",
      "targets": [{
        "expr": "histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m]))",
        "legendFormat": "p99 latency"
      }]
    },
    {
      "title": "Prediction Distribution",
      "type": "graph",
      "targets": [{
        "expr": "rate(model_prediction_value_bucket[1h])",
        "legendFormat": "{{le}}"
      }]
    },
    {
      "title": "Error Rate",
      "type": "singlestat",
      "targets": [{
        "expr": "rate(prediction_errors_total[5m]) / rate(http_requests_total[5m])",
        "legendFormat": "Error Rate"
      }]
    }
  ]
}
```

---

## 12. GitHub Actions — CI/CD for ML

### What It Is

GitHub Actions is the CI/CD platform built into GitHub. Every push to your repo can trigger automated workflows: run tests, validate data, train a model, build a Docker image, and deploy. At work, your team will have GitHub Actions workflows that run on every PR, preventing broken code from being merged.

### Full ML CI/CD Workflow

```yaml
# .github/workflows/ml-pipeline.yml

name: ML CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # ── Job 1: Tests ──────────────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'           # Cache pip — much faster

      - run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint
        run: |
          black --check src/ tests/
          isort --check src/ tests/
          flake8 src/ tests/

      - name: Type check
        run: mypy src/

      - name: Unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  # ── Job 2: Data + Model Tests ─────────────────────────────────────
  model-tests:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - run: pip install -r requirements.txt

      - name: Pull DVC data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          pip install dvc[s3]
          dvc pull data/processed/   # Only pull processed data

      - name: Run DVC pipeline (smoke test with small data)
        run: |
          dvc repro --force --downstream prepare
          # Just run the prepare stage to verify pipeline

      - name: Test feature engineering
        run: pytest tests/test_features.py -v

      - name: Test model prediction
        run: pytest tests/test_model.py -v

  # ── Job 3: Build Docker ───────────────────────────────────────────
  build:
    runs-on: ubuntu-latest
    needs: model-tests
    if: github.ref == 'refs/heads/main'
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GCP Artifact Registry
        uses: docker/login-action@v3
        with:
          registry: us-central1-docker.pkg.dev
          username: _json_key
          password: ${{ secrets.GCP_SA_KEY }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: us-central1-docker.pkg.dev/my-project/ml/churn-model
          tags: |
            type=sha,prefix=git-
            type=semver,pattern={{version}}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha    # Use GitHub Actions cache for layers
          cache-to: type=gha,mode=max

  # ── Job 4: Deploy to Staging ─────────────────────────────────────
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    environment: staging
    steps:
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: churn-model-staging
          image: us-central1-docker.pkg.dev/my-project/ml/churn-model:git-${{ github.sha }}
          region: us-central1
          credentials: ${{ secrets.GCP_SA_KEY }}
          flags: --allow-unauthenticated --max-instances=3

      - name: Smoke test staging
        run: |
          STAGING_URL=$(gcloud run services describe churn-model-staging \
            --region=us-central1 --format='value(status.url)')
          curl -f "${STAGING_URL}/health"
          echo "Staging health check passed ✓"

  # ── Job 5: Deploy to Production (manual approval) ─────────────────
  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production    # Requires approval in GitHub Settings
    steps:
      - name: Deploy to Production
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: churn-model
          image: us-central1-docker.pkg.dev/my-project/ml/churn-model:git-${{ github.sha }}
          region: us-central1
          credentials: ${{ secrets.GCP_SA_KEY }}
          flags: --allow-unauthenticated --min-instances=2 --max-instances=20
```

---

## 13. Poetry — Dependency Management

### What It Is

Poetry is the modern Python dependency management tool. It replaces `pip + requirements.txt + virtual environments` with a single unified workflow. At companies with mature Python practices, Poetry is standard — it gives you reproducible environments, handles dev vs. prod dependencies cleanly, and produces a `poetry.lock` file that pins every transitive dependency.

### Install & Full Usage

```bash
# Install Poetry (once per machine)
curl -sSL https://install.python-poetry.org | python3 -

# New project
poetry new my-ml-project
cd my-ml-project

# Existing project
poetry init   # Creates pyproject.toml interactively

# Add dependencies
poetry add scikit-learn pandas numpy mlflow fastapi uvicorn

# Add dev-only dependencies (tests, linting — NOT in production)
poetry add --group dev pytest black isort mypy flake8 httpx

# Add ML extras
poetry add torch --extras "cpu"       # CPU-only torch (saves 2GB)
poetry add "mlflow[extras]"

# Install everything
poetry install              # Installs all groups
poetry install --only main  # Production: skip dev dependencies

# Run commands in the virtual environment
poetry run python train.py
poetry run pytest tests/
poetry run uvicorn src.app:app

# Activate virtual env in shell
poetry shell

# Export to requirements.txt (for Docker)
poetry export -f requirements.txt --output requirements.txt --without-hashes
poetry export -f requirements.txt --output requirements-dev.txt --with dev

# Update all dependencies to latest compatible versions
poetry update

# Show all installed packages and their versions
poetry show --tree
```

```toml
# pyproject.toml — what Poetry generates/manages
[tool.poetry]
name = "churn-model"
version = "2.1.0"
description = "Customer churn prediction service"
authors = ["Satyam Kumar Jha <satyam@company.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
scikit-learn = "^1.3.2"
pandas = "^2.1.3"
numpy = "^1.26.2"
mlflow = "^2.9.2"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.5.2"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-cov = "^4.1.0"
black = "^23.11.0"
isort = "^5.12.0"
mypy = "^1.7.1"
flake8 = "^6.1.0"
httpx = "^0.25.2"    # For FastAPI testing

[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
warn_unused_configs = true
ignore_missing_imports = true
```

---

## 14. Pre-commit — Code Quality Automation

### What It Is

Pre-commit runs automated checks (formatting, linting, type checking, secret scanning) before every `git commit`. If any check fails, the commit is blocked. This is standard at professional ML teams — it prevents broken, badly formatted, or insecure code from ever entering the repository.

### Install & Setup

```bash
pip install pre-commit
```

```yaml
# .pre-commit-config.yaml — put in repo root

repos:
  # Code formatting (auto-fixes your code)
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.11
        args: [--line-length=100]

  # Import sorting
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  # Linting
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  # General hooks (trailing whitespace, file endings, etc.)
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=500]          # Block files > 500KB (catch accidental data commits)
      - id: detect-private-key       # Block committing private keys!
      - id: check-merge-conflict
      - id: debug-statements         # Block forgotten breakpoint() calls
      - id: no-commit-to-branch      # Protect main branch
        args: [--branch, main]

  # Security scanning — finds hardcoded passwords, API keys
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]

  # Notebook cleaning (remove outputs before commit)
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
```

```bash
# Install hooks (run once per repo)
pre-commit install

# Now every git commit automatically runs all checks
git add . && git commit -m "Add feature engineering"
# black....................................................Passed
# isort....................................................Passed
# flake8...................................................Passed
# check for large files....................................Passed
# detect private key.......................................Passed

# Run manually on all files (useful first time)
pre-commit run --all-files

# Skip pre-commit for an emergency commit (use sparingly!)
git commit -m "Hotfix" --no-verify
```

---

## 15. Papermill — Notebook Execution in Production

### What It Is

Papermill lets you run Jupyter notebooks as parameterized scripts in production. This matters because at many companies, data scientists write analysis in notebooks, and the team wants to automate those notebooks to run nightly with updated data — without converting them to Python scripts. Papermill executes a notebook, injects parameters, and saves the output notebook with all cell outputs included.

### Install

```bash
pip install papermill
```

### Full Working Code

```python
# Execute a notebook programmatically (in Airflow, CI, or scripts)
import papermill as pm
from datetime import datetime

# ── Basic execution ───────────────────────────────────────────────────
pm.execute_notebook(
    input_path="notebooks/weekly_analysis.ipynb",   # Template notebook
    output_path=f"output/analysis_{datetime.now().date()}.ipynb",  # Executed output
    parameters={                                     # Injected into the notebook
        "data_date": str(datetime.now().date()),
        "model_version": "v2.1",
        "threshold": 0.45,
        "output_dir": "reports/"
    },
    kernel_name="python3",
    execution_timeout=1800,     # 30 min timeout
    progress_bar=True
)

# ── Convert output notebook to HTML report ───────────────────────────
import subprocess
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "html",
    "--no-input",                # Hide code cells (for stakeholder reports)
    f"output/analysis_{datetime.now().date()}.ipynb",
    "--output", f"reports/weekly_report_{datetime.now().date()}.html"
])
```

In your notebook, mark the parameters cell with the `parameters` tag:

```python
# Cell tagged "parameters" in Jupyter (Cell → Edit Cell Metadata → add "parameters")
data_date = "2024-01-15"     # Will be overridden by Papermill
model_version = "v1.0"
threshold = 0.45
output_dir = "reports/"
```

```python
# In Airflow DAG:
from airflow.operators.python import PythonOperator

def run_analysis_notebook(**context):
    import papermill as pm
    pm.execute_notebook(
        "notebooks/drift_analysis.ipynb",
        f"/tmp/drift_analysis_{context['ds']}.ipynb",
        parameters={"run_date": context["ds"]}
    )

notebook_task = PythonOperator(
    task_id="run_analysis_notebook",
    python_callable=run_analysis_notebook,
    dag=dag
)
```

---

## Final Reference: Tools by Scenario

| You need to... | Use this tool |
|---|---|
| Track training experiments | **MLflow** or **W&B** |
| Version large datasets | **DVC** |
| Validate data before training | **Great Expectations** |
| Package model for deployment | **Docker** |
| Serve model via REST API | **FastAPI** or **BentoML** |
| Schedule recurring ML jobs | **Airflow** or **Prefect** |
| Monitor data drift in production | **Evidently AI** |
| Monitor service latency & errors | **Prometheus + Grafana** |
| Automate code quality | **pre-commit** |
| Manage Python dependencies | **Poetry** |
| Automate build/test/deploy | **GitHub Actions** |
| Run notebooks in production | **Papermill** |

---

## Your First Week Checklist at an ML Company

```
Day 1-2: Environment Setup
  ✅ Clone repo, set up Poetry environment
  ✅ Install pre-commit hooks: pre-commit install
  ✅ Connect to MLflow / W&B (they'll give you the URL)
  ✅ Configure DVC remote (ask team for credentials)
  ✅ Pull data: dvc pull

Day 3-4: Run Your First Experiment
  ✅ Explore existing notebooks/scripts
  ✅ Run training script, see it log to MLflow UI
  ✅ Modify one hyperparameter, compare runs in MLflow
  ✅ Run the full DVC pipeline: dvc repro

Day 5-7: Understand the Production System
  ✅ Find the Airflow DAG for training (ask team)
  ✅ Find the Grafana dashboard (ask for monitoring URL)
  ✅ Look at Evidently drift reports
  ✅ Read existing GitHub Actions workflows
  ✅ Understand deployment process (Docker → registry → k8s/Cloud Run)
```

---

*Guide version: May 2026 | Tools covered: MLflow, W&B, DVC, Great Expectations, Docker, FastAPI, BentoML, Apache Airflow, Prefect, Evidently AI, Prometheus+Grafana, GitHub Actions, Poetry, Pre-commit, Papermill*
