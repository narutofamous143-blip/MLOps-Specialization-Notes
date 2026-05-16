# MLOps: Complete Beginner to Practitioner Notes
### Everything You Need to Go from Zero to Production ML Systems

---

## Table of Contents

**Part I — Foundation**
1. [What is MLOps?](#chapter-1-what-is-mlops)
2. [The ML Lifecycle](#chapter-2-the-ml-lifecycle)
3. [Why Traditional Software Practices Break in ML](#chapter-3-why-traditional-software-practices-break-in-ml)
4. [MLOps Maturity Levels](#chapter-4-mlops-maturity-levels)

**Part II — Data**

5. [Data Engineering for ML](#chapter-5-data-engineering-for-ml)
6. [Data Versioning with DVC](#chapter-6-data-versioning-with-dvc)
7. [Data Validation & Quality](#chapter-7-data-validation--quality)
8. [Feature Engineering](#chapter-8-feature-engineering)
9. [Feature Stores](#chapter-9-feature-stores)

**Part III — Experimentation**

10. [Experiment Tracking with MLflow](#chapter-10-experiment-tracking-with-mlflow)
11. [Hyperparameter Tuning](#chapter-11-hyperparameter-tuning)
12. [Model Evaluation & Validation](#chapter-12-model-evaluation--validation)

**Part IV — Deployment**

13. [Model Registry & Versioning](#chapter-13-model-registry--versioning)
14. [Model Serving Patterns](#chapter-14-model-serving-patterns)
15. [Model Deployment with FastAPI & BentoML](#chapter-15-model-deployment-with-fastapi--bentoml)
16. [Containerizing ML Models](#chapter-16-containerizing-ml-models)

**Part V — Automation & Orchestration**

17. [CI/CD for Machine Learning](#chapter-17-cicd-for-machine-learning)
18. [ML Pipeline Orchestration](#chapter-18-ml-pipeline-orchestration)
19. [Kubeflow Pipelines](#chapter-19-kubeflow-pipelines)

**Part VI — Monitoring & Reliability**

20. [ML Monitoring — Why Models Degrade](#chapter-20-ml-monitoring--why-models-degrade)
21. [Data Drift & Concept Drift Detection](#chapter-21-data-drift--concept-drift-detection)
22. [Model Performance Monitoring](#chapter-22-model-performance-monitoring)

**Part VII — Cloud MLOps Platforms**

23. [MLOps on Google Cloud (Vertex AI)](#chapter-23-mlops-on-google-cloud-vertex-ai)
24. [MLOps on AWS (SageMaker)](#chapter-24-mlops-on-aws-sagemaker)
25. [MLOps Tool Ecosystem Map](#chapter-25-mlops-tool-ecosystem-map)

---

# PART I — FOUNDATION

---

## Chapter 1: What is MLOps?

Imagine you've spent three weeks training a machine learning model. It achieves 94% accuracy on your test set. You feel great. You hand it to the engineering team to "put it in production." Six months later, the model's accuracy has silently degraded to 71% because the world has changed, but the model hasn't. Nobody noticed until customers started complaining. This scenario plays out at companies every day — and it is precisely the problem MLOps was invented to solve.

**MLOps** stands for **Machine Learning Operations**. It is a set of practices, tools, and cultural philosophies that aim to deploy and maintain machine learning models in production reliably, efficiently, and at scale. Think of it as DevOps — the practice of combining software development with IT operations — but specifically designed for the unique challenges of machine learning systems.

> **Definition**: MLOps is the discipline of applying DevOps principles (automation, monitoring, collaboration, continuous improvement) to the full lifecycle of ML systems — from data collection through model training, deployment, and ongoing monitoring in production.

The term was coined around 2015–2016 as companies like Google, Uber, and Netflix began publishing post-mortems about the extraordinary difficulty of keeping ML models alive in production. Google's famous 2015 paper "Hidden Technical Debt in Machine Learning Systems" described how a trained model is actually a tiny fraction of a real ML system — the surrounding infrastructure for data collection, validation, feature engineering, monitoring, and retraining is vastly larger.

### Why MLOps Matters

According to Gartner, roughly 85% of ML projects never make it into production. Of those that do, a significant fraction fail silently — continuing to serve predictions that are increasingly wrong without anyone realizing it. The root causes are consistently the same: no versioning of data or models, no automated testing, no monitoring, and no automated retraining pipelines.

MLOps addresses this gap by bringing engineering discipline to every phase of the ML lifecycle. When done well, an MLOps practice allows a team to:

- Deploy a model update in hours rather than weeks
- Detect model degradation automatically before it affects users
- Reproduce any past experiment exactly
- Roll back a bad model deployment instantly
- Continuously improve models as new data arrives

### MLOps vs. DevOps vs. DataOps

These three disciplines overlap but are distinct. Understanding the differences helps you understand why ML systems need their own practices.

| Dimension | DevOps | DataOps | MLOps |
|-----------|--------|---------|-------|
| Primary artifact | Software code | Data pipelines | ML models + data + code |
| What changes | Application logic | Data transformations | Model weights + features + data |
| Testing | Unit/integration tests | Data quality tests | Model performance + data tests |
| "Deployment" | Binary or container | ETL pipeline | Model serving endpoint |
| Key failure mode | Broken code | Bad data | Silent model degradation |
| Reproducibility challenge | Code version | Data lineage | Code + data + hyperparameters + environment |

The critical insight is that an ML system has **three things that can change independently**: the code, the data, and the model weights. Traditional DevOps practices only track code changes. MLOps must track all three simultaneously.

---

## Chapter 2: The ML Lifecycle

Before diving into tooling, you must understand the full lifecycle of an ML system. Every tool, practice, and concept in MLOps maps to one or more phases of this lifecycle.

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE ML LIFECYCLE                            │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Data    │───▶│ Feature  │───▶│ Model    │───▶│ Model    │  │
│  │Collection│    │Engineering    │ Training │    │Evaluation│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │         │
│       ▼                                               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Data    │    │Monitoring│◀───│  Model   │◀───│  Model   │  │
│  │Validation│    │& Alerts  │    │ Serving  │    │ Registry │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                       │                                         │
│                       └──────── Retraining Trigger ────────────▶│
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Problem Definition

Every ML project begins with a well-defined problem. This phase is deceptively important — teams that skip it often build technically impressive models that solve the wrong problem. A good problem definition answers: What decision are we automating? What data do we have? What does success look like in business terms (not just model accuracy)?

> **Example**: "Reduce customer churn" is a business goal, not an ML problem. The ML problem is: "Predict, 30 days in advance, which customers have >70% probability of cancelling their subscription, so the retention team can intervene." This definition specifies the prediction horizon, the output type (probability), and the downstream use case.

### Phase 2: Data Collection and Ingestion

Data is the foundation of any ML system. In this phase, raw data is collected from various sources — databases, APIs, user interactions, sensors, third-party providers — and ingested into a centralized store (data lake or data warehouse). The quality of your data entirely determines the ceiling of your model's performance.

Key activities include identifying data sources, establishing data pipelines, handling missing data policies, and understanding data freshness requirements (how recent must the data be?).

### Phase 3: Data Validation and Exploration

Once data is ingested, it must be validated. This means checking that data conforms to expected schemas, that value distributions match expectations, and that there are no anomalies (sudden spikes, missing columns, unexpected categories) that could silently corrupt training.

Exploratory Data Analysis (EDA) follows — statistical summaries, correlation analysis, visualization of distributions — to build intuition about the data before any modeling begins.

### Phase 4: Feature Engineering

Raw data is rarely in a form that ML algorithms can consume directly. Feature engineering transforms raw data into **features** — numerical representations that capture the signal relevant to the prediction task. This is often the most impactful phase of the ML lifecycle; a simple model trained on excellent features will consistently outperform a complex model trained on poor features.

Examples include: converting timestamps into "day of week" and "hour of day"; encoding categorical variables; creating interaction terms; normalizing numerical ranges; computing rolling aggregates over time windows.

### Phase 5: Model Training

With features prepared, you select an algorithm (or multiple algorithms) and train the model by optimizing its parameters on the training data. This phase includes experiment tracking (recording which hyperparameters, data versions, and code produced which results), hyperparameter tuning, and cross-validation.

### Phase 6: Model Evaluation

A trained model must be rigorously evaluated before deployment. This goes beyond a single accuracy number — it includes fairness analysis (does the model perform equally well across demographic groups?), robustness testing (how does it perform on edge cases?), and business metric validation (does improving the ML metric actually improve the business outcome?).

### Phase 7: Model Registry and Versioning

Approved models are stored in a **Model Registry** — a centralized catalog that tracks all model versions, their metadata (training date, dataset used, performance metrics), and their lifecycle stage (Staging, Production, Archived). This is the source of truth for "which model is currently serving predictions."

### Phase 8: Model Deployment and Serving

Deployment makes the model available for inference. There are several patterns: batch inference (run predictions overnight on a dataset), real-time online inference (an API endpoint that responds within milliseconds), and streaming inference (predictions on events as they arrive). Each pattern has different infrastructure requirements.

### Phase 9: Monitoring and Observability

This is where most teams underinvest and pay the price later. Once a model is in production, it must be continuously monitored — not just for system health (latency, error rates) but for **model health** (are the predictions still accurate? has the input data distribution changed?). Monitoring feeds back into the lifecycle, triggering retraining when model performance degrades.

---

## Chapter 3: Why Traditional Software Practices Break in ML

Understanding where traditional software engineering practices fail in ML is not academic — it directly explains every tool and practice in MLOps. There are five fundamental differences.

### Difference 1: Code + Data = Behavior (not just code)

In traditional software, behavior is entirely determined by code. If you version your code, you can reproduce any past behavior. In ML, behavior is determined by **both code and data**. Two runs of identical code on slightly different data produce different models with different behaviors. This means version control for code alone is insufficient — you must also version data.

### Difference 2: ML Systems Have Emergent Behavior

You cannot reason about a trained neural network the way you reason about a for-loop. Given inputs and a decision tree, you can trace exactly why a prediction was made. Given inputs and a deep learning model with millions of parameters, you often cannot. This makes testing ML systems fundamentally different — you cannot write a unit test that says "given input X, the model will return exactly Y." You can only say "given a test set, the model should have accuracy ≥ Z."

### Difference 3: The Environment is a Moving Target

Traditional software runs in a fixed environment. A sorting function sorts the same way today as it did last year. ML models are trained on historical data and then deployed to make predictions about the future. The future inevitably differs from the past — this is called **distribution shift** or **data drift** — and models degrade gracefully (or not) as the gap widens.

> **Example**: A fraud detection model trained on 2022 transaction data learns that purchases of >$500 from foreign merchants are suspicious. In 2023, remote work becomes ubiquitous and cross-border purchases increase dramatically. The model now flags legitimate transactions as fraud. The code didn't change. The data didn't change. The *world* changed.

### Difference 4: Experiments are First-Class Citizens

Software development is a relatively linear process: write code, test, deploy. ML development is highly exploratory: run 50 experiments with different features and algorithms, compare them, pick the winner, run 50 more. If experiments are not tracked systematically, reproducibility is lost. Teams routinely find themselves unable to reproduce their best model from two months ago because they didn't record which exact dataset version, random seed, and hyperparameters produced it.

### Difference 5: Training-Serving Skew

The code that computes features during training is often written separately from the code that computes features during serving (inference). If these two codepaths diverge — even slightly — the model sees different data in production than it was trained on. This is called **training-serving skew** and it is one of the most common and subtle failure modes in ML systems.

> **Example**: During training, your feature pipeline computes "days since last purchase" using `(training_date - last_purchase_date)`. During serving, a bug computes it as `(last_purchase_date - serving_date)` producing a negative number. The model was never trained on negative values for this feature and makes erratic predictions. The model itself is fine; the pipeline is broken.

---

## Chapter 4: MLOps Maturity Levels

Teams don't implement all of MLOps at once. Google defined a widely-used maturity model with three levels that describes the progression from ad hoc experimentation to fully automated ML systems.

### Level 0: Manual Process (No MLOps)

This is where most teams start. Experiments are run in Jupyter notebooks. There is no systematic tracking of experiments. Data is processed in ad hoc scripts. Deployment is manual — a data scientist exports a `.pkl` file, emails it to a software engineer, and hopes for the best. Retraining happens whenever someone remembers to do it.

**Characteristics:**
- No versioning of data, code, or models
- No automated testing of models or data pipelines
- Deployment is a manual, one-time event
- No monitoring of model performance in production
- Retraining is manual and infrequent

**Pain points**: Cannot reproduce experiments; deployment takes weeks; model performance degrades unnoticed; knowledge is siloed in individual notebooks.

### Level 1: ML Pipeline Automation

The training process is automated into a pipeline that can be triggered on demand. Data preprocessing, feature engineering, training, and evaluation steps are formalized as code. Experiment tracking is introduced. Models are stored in a registry. Deployment is still somewhat manual but follows a defined process.

**Characteristics:**
- Training pipeline is automated end-to-end
- Experiment tracking with tools like MLflow
- Model registry for versioning
- Continuous Training (CT): the pipeline can be triggered by new data
- Basic monitoring is in place

**Improvement over Level 0**: A new model can be trained and deployed in hours rather than weeks. Experiments are reproducible. Models have proper versioning.

### Level 2: CI/CD Pipeline Automation

The full software engineering CI/CD discipline is applied to ML. Every change to code, data, or model triggers automated testing and validation pipelines. Model updates are deployed automatically when they pass quality gates. The entire system is observable and self-healing.

**Characteristics:**
- Automated testing of data pipelines, features, and model performance
- Continuous Integration: every code change triggers automated training and evaluation
- Continuous Deployment: models that pass quality gates are automatically deployed
- Continuous Monitoring with automated alerts and retraining triggers
- Full data and model lineage tracking

**This is the target state** for production ML systems at scale. However, it represents significant engineering investment and is appropriate only when the business value of the ML system justifies it.

```
MATURITY PROGRESSION:

Level 0           Level 1              Level 2
─────────────     ────────────────     ──────────────────────
Manual scripts    Automated pipeline   Full CI/CD automation
No tracking       MLflow tracking      Automated quality gates
No versioning     Model registry       End-to-end lineage
Manual deploy     Semi-auto deploy     Auto deploy on pass
No monitoring     Basic monitoring     Drift detection + auto-retrain
```

---

# PART II — DATA

---

## Chapter 5: Data Engineering for ML

Data engineering for ML is different from general data engineering. A data warehouse for business intelligence needs to be accurate and queryable. A data pipeline for ML needs all of that — plus **reproducibility**, **point-in-time correctness**, and **feature consistency** between training and serving.

### 5.1 Data Sources and Ingestion

ML data comes from many sources, each with different characteristics and challenges.

**Structured data** (databases, CSVs, data warehouses) is the most common. It is tabular, well-defined, and relatively easy to work with. The challenge is handling schema evolution — when columns are added, renamed, or removed over time.

**Unstructured data** (images, text, audio, video) requires different storage strategies. Large binary files should live in object storage (S3, GCS) with metadata in a database. Never store large binary blobs in relational databases.

**Streaming data** (event streams from Kafka, Pub/Sub) presents a special challenge: you need to train on historical batches but serve predictions on real-time streams. The feature computation logic must work in both batch and streaming modes consistently.

**Third-party data** (purchased datasets, API data) introduces dependencies that can change without notice. Always maintain your own copy rather than fetching from external sources at training time.

### 5.2 The Data Lake vs. Data Warehouse Architecture

Understanding the difference between these two storage paradigms is essential for ML practitioners.

A **data lake** stores raw, unprocessed data in its native format. It is a large, cheap storage repository (like S3 or GCS) where you dump everything and figure out structure later. The philosophy is "store now, query later." Data lakes are ideal as the landing zone for all raw data before it is processed for ML.

A **data warehouse** stores processed, structured data organized for query performance. Systems like BigQuery, Snowflake, and Redshift apply schema-on-write — data must conform to a schema before it can be loaded. Warehouses are ideal for feature computation from structured data.

For ML, the modern architecture is a **Lakehouse** — a data lake with metadata and ACID transaction capabilities added on top (Apache Delta Lake, Apache Iceberg, Apache Hudi). This gives you the cheap storage of a lake with the query performance and reliability of a warehouse.

### 5.3 Data Pipelines for ML

A data pipeline is a sequence of transformations that takes raw data and produces training-ready datasets. For ML, pipelines must have specific properties.

**Idempotency**: Running the pipeline twice on the same input should produce the same output. This enables safe reruns when a step fails partway through.

**Determinism**: Given the same input and code version, the pipeline should produce identical output. Avoid using `datetime.now()` or random seeds that change between runs without explicit parameterization.

**Point-in-time correctness**: When training a model to predict what happens at time T, all features used as inputs must reflect information available before time T. Using information from after T is called **data leakage** and produces models that appear excellent in training but fail catastrophically in production.

> **Example of Data Leakage**: You're training a model to predict whether a loan will default. One of your features is "total number of late payments the customer ever made." If this count includes late payments that occurred after the loan was issued, you've leaked the future into the training data. The model learns a trivial pattern ("people who later made late payments defaulted") that isn't useful for prediction.

```python
# WRONG: Data leakage — uses future information
def create_features_wrong(df):
    df['total_late_payments'] = df.groupby('customer_id')['is_late'].transform('sum')
    # This includes ALL late payments, including future ones!
    return df

# CORRECT: Point-in-time correct feature
def create_features_correct(df, prediction_date):
    # Only count late payments BEFORE the prediction date
    past_payments = df[df['payment_date'] < prediction_date]
    late_counts = past_payments.groupby('customer_id')['is_late'].sum().reset_index()
    late_counts.columns = ['customer_id', 'total_late_payments_to_date']
    return df.merge(late_counts, on='customer_id', how='left')
```

### 5.4 Train/Validation/Test Split Strategy

How you split data profoundly affects whether your model evaluation is trustworthy.

**Random split** is appropriate for i.i.d. (independent, identically distributed) data where there is no time dependency. Shuffle the dataset and take 70% for training, 15% for validation, 15% for testing.

**Time-based split** is essential for time-series data and any data where temporal order matters. You must train on the past and evaluate on the future — never the reverse. The test set must be the most recent data.

```
Time-based split (correct for time-series):
──────────────────────────────────────────────────────►  time
│◄── Training (70%) ──►│◄── Val (15%) ──►│◄── Test (15%) ──►│

Random split (wrong for time-series):
Would mix future data into training — creates leakage!
```

**Stratified split** ensures that class proportions are preserved in each split. For a fraud detection dataset where only 1% of transactions are fraudulent, a random split might put all fraud cases in test by chance. Stratified split guarantees each split contains approximately 1% fraud.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Stratified split (preserves class proportions)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # This is the key parameter
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"Train size: {len(X_train)}, Fraud rate: {y_train.mean():.3f}")
print(f"Val size: {len(X_val)}, Fraud rate: {y_val.mean():.3f}")
print(f"Test size: {len(X_test)}, Fraud rate: {y_test.mean():.3f}")
```

---

## Chapter 6: Data Versioning with DVC

### 6.1 The Problem DVC Solves

Git is excellent at versioning text files (code). It is terrible at versioning large binary files (datasets, model weights). A 10GB training dataset committed to Git will make the repository unusably slow and expensive. **DVC (Data Version Control)** solves this by separating metadata (tracked in Git) from actual data (stored in remote storage like S3, GCS, or Azure Blob).

> **Mental model**: DVC uses Git to track *pointers* to data, while the actual data lives in remote storage. When you do `dvc pull`, DVC reads the pointer from Git, then fetches the actual file from remote storage. This means your Git repository stays small while your data is properly versioned.

### 6.2 DVC Installation and Setup

```bash
# Install DVC with your storage backend
pip install dvc[s3]    # for AWS S3
pip install dvc[gs]    # for Google Cloud Storage
pip install dvc[azure] # for Azure Blob Storage
pip install dvc        # just local storage (for learning)

# Initialize DVC in your project (must be a Git repo)
git init my-ml-project
cd my-ml-project
dvc init

# This creates a .dvc/ directory and .dvcignore
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### 6.3 Tracking Data Files

```bash
# Add a dataset to DVC tracking
dvc add data/raw/customers.csv

# This creates data/raw/customers.csv.dvc — a small metadata file
# containing the MD5 hash and size of the file
cat data/raw/customers.csv.dvc
# outs:
# - md5: a1b2c3d4e5f6...
#   size: 104857600
#   path: customers.csv

# The actual data is now in .dvc/cache/
# Add the .dvc file to Git (NOT the actual data)
git add data/raw/customers.csv.dvc data/raw/.gitignore
git commit -m "Add customer dataset v1"

# Configure remote storage (where actual data is stored)
dvc remote add -d myremote s3://my-ml-bucket/dvcstore
git add .dvc/config
git commit -m "Configure DVC remote storage"

# Push data to remote
dvc push

# Pull data on another machine (after git clone)
git clone https://github.com/my-org/my-ml-project
cd my-ml-project
dvc pull  # Downloads data from S3
```

### 6.4 DVC Pipelines — The Core of Reproducible ML

DVC's most powerful feature is **pipelines** — a way to define your entire data processing and training workflow as a DAG (Directed Acyclic Graph) of stages. DVC tracks the dependencies (input files, parameters) and outputs of each stage, and only re-runs stages when their dependencies change.

```yaml
# dvc.yaml — defines your ML pipeline
stages:
  prepare:
    cmd: python src/prepare.py --input data/raw/customers.csv --output data/processed/
    deps:
      - src/prepare.py
      - data/raw/customers.csv
    params:
      - prepare.test_size
      - prepare.random_seed
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/processed/train.csv
    outs:
      - data/features/train_features.csv
      - data/features/test_features.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features/train_features.csv
    params:
      - train.n_estimators
      - train.max_depth
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/features/test_features.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

```yaml
# params.yaml — all hyperparameters in one file
prepare:
  test_size: 0.2
  random_seed: 42

train:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
```

```bash
# Run the full pipeline
dvc repro

# DVC checks each stage's dependencies. If nothing changed, stages are skipped.
# Stage 'prepare'... cached
# Stage 'featurize'... running
# Stage 'train'... running

# Compare metrics between experiments
dvc metrics show
dvc metrics diff HEAD~1  # Compare current metrics to previous commit

# Visualize the pipeline DAG
dvc dag
```

### 6.5 DVC Experiments — Lightweight Experiment Management

```bash
# Run an experiment with different hyperparameters
dvc exp run --set-param train.n_estimators=200 --set-param train.max_depth=8

# Run multiple experiments (grid search style)
dvc exp run --set-param train.n_estimators=100,200,300

# Show all experiments
dvc exp show

# Compare experiments
dvc exp diff exp-a3b2c1 exp-d4e5f6

# Apply the best experiment (make it the new HEAD)
dvc exp apply exp-a3b2c1
```

---

## Chapter 7: Data Validation & Quality

### 7.1 Why Data Validation is Non-Negotiable

Garbage in, garbage out. This maxim is especially true in ML, where bad data doesn't cause an immediate crash — it causes a model that silently learns the wrong patterns. Data validation is the automated check that catches bad data before it contaminates your model.

There are two distinct places where data validation must happen: at **training time** (validating the dataset before training begins) and at **serving time** (validating incoming requests before they're sent to the model). Both are essential.

### 7.2 Great Expectations — Data Validation Framework

**Great Expectations (GX)** is the industry-standard Python library for data validation. The core concept is an **Expectation** — a declarative statement about what your data should look like (e.g., "the `age` column should never be negative"). Expectations are grouped into **Expectation Suites**, run against actual data in a **Validation**, and results are stored in a **Data Docs** site.

```python
import great_expectations as gx

# Initialize a Data Context
context = gx.get_context()

# Create a datasource pointing to your data
datasource = context.sources.add_pandas_filesystem(
    name="my_data",
    base_directory="data/raw/"
)
asset = datasource.add_csv_asset(name="customers", batching_regex="customers.csv")
batch_request = asset.build_batch_request()

# Create an expectation suite
suite = context.add_expectation_suite("customer_data_suite")

# Define expectations
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="customer_data_suite"
)

# Column existence and type checks
validator.expect_column_to_exist("customer_id")
validator.expect_column_to_exist("age")
validator.expect_column_to_exist("signup_date")
validator.expect_column_to_exist("churn")

# Value range checks
validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)
validator.expect_column_values_to_be_between("purchase_amount", min_value=0)

# Null checks
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_not_be_null("churn")

# Uniqueness
validator.expect_column_values_to_be_unique("customer_id")

# Categorical value checks
validator.expect_column_values_to_be_in_set(
    "plan_type",
    value_set=["free", "basic", "premium", "enterprise"]
)

# Statistical distribution checks
validator.expect_column_mean_to_be_between("age", min_value=25, max_value=55)
validator.expect_column_stdev_to_be_between("purchase_amount", min_value=10, max_value=500)

# Row count check
validator.expect_table_row_count_to_be_between(min_value=10000, max_value=10000000)

# Save and run validation
validator.save_expectation_suite(discard_failed_expectations=False)

checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validations=[{"batch_request": batch_request,
                  "expectation_suite_name": "customer_data_suite"}]
)

results = checkpoint.run()
print(f"Validation passed: {results.success}")
```

### 7.3 TensorFlow Data Validation (TFDV)

For ML-specific data validation — especially detecting statistical drift between training and serving data — **TensorFlow Data Validation (TFDV)** is highly useful.

```python
import tensorflow_data_validation as tfdv
import pandas as pd

# Compute statistics from training data
train_df = pd.read_csv("data/processed/train.csv")
train_stats = tfdv.generate_statistics_from_dataframe(train_df)

# Infer schema from statistics
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)

# Later: validate new data against the training schema
new_data_df = pd.read_csv("data/new_batch/latest.csv")
new_stats = tfdv.generate_statistics_from_dataframe(new_data_df)

# Find anomalies
anomalies = tfdv.validate_statistics(
    statistics=new_stats,
    schema=schema
)
tfdv.display_anomalies(anomalies)
# Output might show:
# Feature 'plan_type': New value 'enterprise_plus' not in schema
# Feature 'age': Fraction of values in range [18, 120] is 0.94, min is 0.95

# Compare training vs serving statistics visually
tfdv.visualize_statistics(
    lhs_statistics=train_stats,
    rhs_statistics=new_stats,
    lhs_name="Training Data",
    rhs_name="New Serving Data"
)
```

---

## Chapter 8: Feature Engineering

### 8.1 What Features Are and Why They Matter

A **feature** is a measurable property of the phenomenon you're trying to model. Raw data (a user's signup timestamp) is not directly useful as a feature. Engineered features derived from that timestamp (day of week, hour of day, days since signup, is the user in their first 7 days?) capture signal that the model can learn from.

The quality of features determines the ceiling of your model's performance far more than the choice of algorithm. This observation is so consistent that it has become a fundamental principle: **feature engineering is the highest-leverage activity in applied ML**.

### 8.2 Common Feature Engineering Techniques

**Numerical Features:**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

df = pd.read_csv("data/customers.csv")

# Log transformation (for right-skewed distributions like income, order value)
df['log_purchase_amount'] = np.log1p(df['purchase_amount'])  # log1p handles zeros

# Binning continuous variables (age groups)
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['18-25', '26-35', '36-45', '46-55', '55+']
)

# Interaction features (product of two related features)
df['revenue_per_session'] = df['total_revenue'] / (df['session_count'] + 1)

# Normalization (scale to [0, 1]) — use for distance-based algorithms
scaler = MinMaxScaler()
df['age_normalized'] = scaler.fit_transform(df[['age']])

# Standardization (zero mean, unit variance) — use for linear models, neural nets
std_scaler = StandardScaler()
df['purchase_standardized'] = std_scaler.fit_transform(df[['purchase_amount']])

# Box-Cox / Yeo-Johnson for Gaussianizing skewed data
pt = PowerTransformer(method='yeo-johnson')
df['purchase_transformed'] = pt.fit_transform(df[['purchase_amount']])
```

**Categorical Features:**

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd

# One-Hot Encoding (for nominal categories with no inherent order)
# Use when category count is small (< ~20 categories)
df_encoded = pd.get_dummies(df, columns=['plan_type', 'country'], drop_first=True)

# Ordinal Encoding (for categories with natural order)
size_mapping = {'small': 1, 'medium': 2, 'large': 3, 'enterprise': 4}
df['company_size_encoded'] = df['company_size'].map(size_mapping)

# Target Encoding (encode category by its mean target value)
# Reduces dimensionality for high-cardinality categoricals
# WARNING: causes leakage if not done properly (compute on train only)
train_means = df_train.groupby('country')['churn'].mean()
df_train['country_encoded'] = df_train['country'].map(train_means)
df_test['country_encoded'] = df_test['country'].map(train_means)
# Fill unseen countries with global mean
df_test['country_encoded'].fillna(df_train['churn'].mean(), inplace=True)

# Frequency Encoding (encode by how often the category appears)
freq = df['city'].value_counts() / len(df)
df['city_frequency'] = df['city'].map(freq)
```

**Datetime Features:**

```python
import pandas as pd

df['signup_date'] = pd.to_datetime(df['signup_date'])

# Extract temporal components
df['signup_year'] = df['signup_date'].dt.year
df['signup_month'] = df['signup_date'].dt.month
df['signup_day_of_week'] = df['signup_date'].dt.dayofweek  # 0=Monday
df['signup_hour'] = df['signup_date'].dt.hour
df['signup_is_weekend'] = df['signup_date'].dt.dayofweek >= 5

# Cyclic encoding (preserves the circular nature of time)
# Monday (0) and Sunday (6) are "close" but raw encoding shows them as far apart
import numpy as np
df['day_sin'] = np.sin(2 * np.pi * df['signup_day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['signup_day_of_week'] / 7)

df['month_sin'] = np.sin(2 * np.pi * df['signup_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['signup_month'] / 12)

# Time since a reference date
df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days
```

**Window/Aggregation Features (for time-series):**

```python
import pandas as pd

# Sort by time before any window operation
df = df.sort_values(['customer_id', 'transaction_date'])

# Rolling aggregates
df['purchase_7d_sum'] = df.groupby('customer_id')['purchase_amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).sum()
)
df['purchase_30d_mean'] = df.groupby('customer_id')['purchase_amount'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)
df['purchase_7d_std'] = df.groupby('customer_id')['purchase_amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
)

# Lag features (value at previous time step)
df['purchase_prev_day'] = df.groupby('customer_id')['purchase_amount'].shift(1)
df['purchase_lag_7d'] = df.groupby('customer_id')['purchase_amount'].shift(7)

# Trend feature (is the customer spending more or less?)
df['purchase_trend_7d'] = df['purchase_7d_sum'] - df.groupby('customer_id')['purchase_amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).sum().shift(7)
)
```

### 8.3 Feature Selection

Not all features improve model performance. Including too many features can cause **overfitting** (the model memorizes training data instead of learning generalizable patterns) and increases computational cost. Feature selection identifies the most predictive subset.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv("data/features/train_features.csv")
y_train = pd.read_csv("data/features/train_labels.csv").squeeze()

# Method 1: Feature importance from tree models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(20))

# Select features above a threshold
selector = SelectFromModel(rf, threshold='median')
X_train_selected = selector.transform(X_train)
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} features from {X_train.shape[1]}")

# Method 2: Correlation-based removal (remove highly correlated pairs)
corr_matrix = X_train.corr().abs()
upper_triangle = corr_matrix.where(
    pd.DataFrame([[True if i < j else False
                   for j in range(len(corr_matrix.columns))]
                  for i in range(len(corr_matrix.columns))],
                 columns=corr_matrix.columns, index=corr_matrix.index)
)
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
print(f"Dropping highly correlated features: {to_drop}")
X_train_uncorr = X_train.drop(columns=to_drop)
```

---

## Chapter 9: Feature Stores

### 9.1 The Problem Feature Stores Solve

As an organization builds more ML models, a serious problem emerges: the same features get recomputed independently by multiple teams. Team A computes "average purchase value in last 30 days" for their recommendation model. Team B computes the same feature (slightly differently) for their fraud model. Team C computes a third variation. You now have three implementations that should be identical but differ subtly — and any one of them could be wrong.

Worse, each team must maintain their own serving infrastructure to compute these features in real time. And when a new dataset arrives, all three teams must update their pipelines independently.

A **Feature Store** solves this by providing a centralized platform for:
1. **Computing** features once, using shared definitions
2. **Storing** feature values for fast retrieval at both training and serving time
3. **Sharing** features across teams and models
4. **Ensuring consistency** between training and serving computations

### 9.2 Architecture of a Feature Store

```
┌──────────────────────────────────────────────────────────────┐
│                    FEATURE STORE ARCHITECTURE                │
│                                                              │
│  Data Sources ──► Feature Pipeline ──► Feature Store        │
│  (DB, S3, Kafka)    (Spark, Beam)     ┌──────────────────┐  │
│                                       │  Online Store    │  │
│                                       │  (Redis/DynamoDB)│  │
│                                       │  Low latency <5ms│  │
│                                       ├──────────────────┤  │
│                                       │  Offline Store   │  │
│                                       │  (S3/BigQuery)   │  │
│                                       │  Historical data │  │
│                                       └──────────────────┘  │
│                                              │  │            │
│  Training ◄─────────────────────────────────── │            │
│  (gets historical features for a time window)  │            │
│                                                │            │
│  Serving ◄─────────────────────────────────────┘            │
│  (gets latest feature values, <5ms latency)                 │
└──────────────────────────────────────────────────────────────┘
```

The key architectural insight is the **dual storage approach**: features are stored in both an **offline store** (cheap, slow, historical — for training) and an **online store** (expensive, fast, latest values — for serving). The same feature definition generates values in both stores.

### 9.3 Feast — Open Source Feature Store

**Feast** is the most widely used open-source feature store. It integrates with most major data systems and cloud providers.

```bash
pip install feast[gcp]  # or feast[aws] or feast[redis]
feast init my_feature_repo
cd my_feature_repo
```

```python
# feature_repo/features.py — Define features declaratively

from datetime import timedelta
from feast import (
    Entity, Feature, FeatureView, FileSource, ValueType,
    FeatureService, Field
)
from feast.types import Float32, Float64, Int64, String

# Entity: the primary key that features are associated with
customer = Entity(
    name="customer_id",
    description="Customer identifier",
    value_type=ValueType.STRING
)

# Data source: where raw feature data lives
customer_stats_source = FileSource(
    path="data/features/customer_stats.parquet",
    timestamp_field="event_timestamp",  # Point-in-time correctness
    created_timestamp_column="created",
)

# Feature view: a group of features computed from a source
customer_stats_fv = FeatureView(
    name="customer_stats",
    entities=["customer_id"],
    ttl=timedelta(days=30),  # How long features remain valid
    schema=[
        Field(name="purchase_count_30d", dtype=Int64),
        Field(name="avg_purchase_amount_30d", dtype=Float64),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="total_sessions_7d", dtype=Int64),
        Field(name="churn_risk_score", dtype=Float32),
    ],
    source=customer_stats_source,
    online=True,  # Store in online store for real-time serving
)

# Feature service: a named set of features for a specific model
churn_model_features = FeatureService(
    name="churn_prediction_v1",
    features=[
        customer_stats_fv[["purchase_count_30d",
                            "avg_purchase_amount_30d",
                            "days_since_last_purchase"]],
    ],
    description="Features for the churn prediction model"
)
```

```python
# feast_operations.py — Using the feature store

from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="feature_repo/")

# TRAINING: Get historical features (point-in-time correct)
entity_df = pd.DataFrame({
    "customer_id": ["c001", "c002", "c003"],
    # timestamp defines WHEN we want the feature values from
    "event_timestamp": [
        datetime(2024, 1, 15),
        datetime(2024, 1, 15),
        datetime(2024, 2, 1),
    ]
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_stats:purchase_count_30d",
              "customer_stats:avg_purchase_amount_30d",
              "customer_stats:days_since_last_purchase"]
).to_df()
print(training_df)

# SERVING: Get latest feature values (real-time, from online store)
# First, materialize features into the online store
store.materialize_incremental(end_date=datetime.now())

# Then retrieve at serving time
online_features = store.get_online_features(
    features=["customer_stats:purchase_count_30d",
              "customer_stats:avg_purchase_amount_30d"],
    entity_rows=[
        {"customer_id": "c001"},
        {"customer_id": "c002"},
    ]
).to_dict()

print(online_features)
# {'customer_id': ['c001', 'c002'],
#  'purchase_count_30d': [42, 17],
#  'avg_purchase_amount_30d': [89.50, 234.20]}
```

```bash
# Apply feature definitions to the store
feast apply

# Materialize features into online store (backfill)
feast materialize 2024-01-01T00:00:00 2024-12-31T00:00:00

# Incrementally materialize new data
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

# PART III — EXPERIMENTATION

---

## Chapter 10: Experiment Tracking with MLflow

### 10.1 The Experiment Tracking Problem

Without systematic experiment tracking, ML development becomes a graveyard of lost knowledge. You run an experiment, get a good result, move on. Three weeks later you want to reproduce it — but you've changed the code, can't remember which dataset split you used, and the hyperparameters were only in a Jupyter cell you've since overwritten. This is not an edge case; it is the default state of unstructured ML development.

**Experiment tracking** is the practice of systematically recording every detail of every training run: the code version, data version, hyperparameters, environment, and resulting metrics. With proper tracking, you can always answer: "What exactly produced this model, and how do I recreate it?"

**MLflow** is the dominant open-source experiment tracking platform. It has four components that work independently or together: Tracking, Projects, Models, and Registry.

### 10.2 MLflow Tracking — Recording Experiments

```bash
pip install mlflow scikit-learn pandas

# Start the MLflow UI (local server)
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

```python
# train_with_mlflow.py — A complete training script with MLflow tracking

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Configure tracking server (local or remote)
mlflow.set_tracking_uri("http://localhost:5000")
# For remote: mlflow.set_tracking_uri("postgresql://user:pass@host/db")

# Set the experiment (creates it if it doesn't exist)
mlflow.set_experiment("churn-prediction")

# Load data
X_train = pd.read_csv("data/features/train_features.csv")
y_train = pd.read_csv("data/features/train_labels.csv").squeeze()
X_test = pd.read_csv("data/features/test_features.csv")
y_test = pd.read_csv("data/features/test_labels.csv").squeeze()

# Define hyperparameters
params = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_split": 20,
    "random_state": 42
}

# Start an MLflow run
with mlflow.start_run(run_name="GBM-experiment-001") as run:
    print(f"Run ID: {run.info.run_id}")

    # Log all hyperparameters at once
    mlflow.log_params(params)

    # Log the data version (use git hash or DVC hash)
    mlflow.log_param("train_data_version", "v2.3.1")
    mlflow.log_param("n_train_samples", len(X_train))
    mlflow.log_param("n_features", X_train.shape[1])

    # Log tags (useful for filtering experiments later)
    mlflow.set_tags({
        "model_type": "gradient_boosting",
        "dataset": "customer_churn_2024",
        "author": "satyam",
        "purpose": "production_candidate"
    })

    # Train the model
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
    mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Log all metrics
    mlflow.log_metrics(metrics)

    # Log metrics at each training step (for learning curves)
    # MLflow supports "step" for time-series metrics
    for step, train_loss in enumerate(model.train_score_):
        mlflow.log_metric("train_loss", train_loss, step=step)

    # Create and log a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()

    # Log feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    feature_importance.plot(kind='barh', x='feature', y='importance', ax=ax)
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()

    # Log the model with its input schema (signature)
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, y_pred_proba)
    input_example = X_train.iloc[:5]

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="churn-prediction-model"  # Auto-register
    )

    # Log any additional files (feature list, preprocessing config)
    feature_list = X_train.columns.tolist()
    with open("artifacts/feature_list.json", "w") as f:
        json.dump(feature_list, f)
    mlflow.log_artifact("artifacts/feature_list.json")

    print(f"Metrics: {metrics}")
    print(f"Model logged to run: {run.info.run_id}")
```

### 10.3 Querying MLflow Programmatically

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

# Find the best run in an experiment
experiment = client.get_experiment_by_name("churn-prediction")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.roc_auc > 0.85 AND params.model_type = 'gradient_boosting'",
    order_by=["metrics.roc_auc DESC"],
    max_results=10
)

best_run = runs[0]
print(f"Best run: {best_run.info.run_id}")
print(f"ROC AUC: {best_run.data.metrics['roc_auc']:.4f}")
print(f"Params: {best_run.data.params}")

# Load model from best run
model_uri = f"runs:/{best_run.info.run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
predictions = loaded_model.predict(X_test)
```

### 10.4 MLflow Projects — Reproducible Runs

An MLflow Project is a convention for packaging ML code so it can be run reproducibly by anyone.

```yaml
# MLproject file at repo root
name: churn-prediction

conda_env: conda.yaml
# OR python_env: python_env.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
      learning_rate: {type: float, default: 0.1}
      data_version: {type: str, default: "v2.3"}
    command: "python train_with_mlflow.py --n-estimators {n_estimators} --max-depth {max_depth} --lr {learning_rate}"

  preprocess:
    parameters:
      input_path: {type: str}
      output_path: {type: str, default: "data/processed"}
    command: "python src/preprocess.py {input_path} {output_path}"
```

```bash
# Run project locally
mlflow run . -P n_estimators=200 -P max_depth=6

# Run project from GitHub (any git ref)
mlflow run https://github.com/my-org/churn-model -P n_estimators=300

# Run project on remote compute (Kubernetes)
mlflow run . --backend kubernetes --backend-config k8s_config.json
```

---

## Chapter 11: Hyperparameter Tuning

### 11.1 What Are Hyperparameters?

Model parameters are values the model *learns* from data (like the weights of a neural network). **Hyperparameters** are settings that control the *learning process itself* and must be chosen before training begins. Examples include learning rate, number of trees, tree depth, regularization strength, dropout rate, and batch size.

Choosing hyperparameters poorly leads to models that are either **underfitting** (too simple — high bias, poor performance on both train and test) or **overfitting** (too complex — low training loss but high test loss). Good hyperparameter selection is often the difference between a mediocre model and a state-of-the-art one.

### 11.2 Search Strategies

**Grid Search**: Exhaustively tries every combination of specified values. Reliable but exponentially expensive. Use only when the hyperparameter space is very small.

**Random Search**: Randomly samples hyperparameter combinations. Surprisingly more efficient than grid search for high-dimensional spaces — research shows random search often finds equally good solutions with far fewer trials.

**Bayesian Optimization**: Builds a probabilistic model of the objective function and uses it to choose which hyperparameters to try next, focusing search on promising regions. **Optuna** is the leading library for this approach.

```python
import optuna
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Optuna objective function — called once per trial."""

    # Define the search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": 42
    }

    model = GradientBoostingClassifier(**params)

    # 5-fold CV to estimate generalization performance
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='roc_auc', n_jobs=-1
    )

    # Log each trial to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_roc_auc", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

    # Report for pruning (Optuna can stop unpromising trials early)
    trial.report(cv_scores.mean(), step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return cv_scores.mean()

# Create and run a study
with mlflow.start_run(run_name="optuna-hyperparameter-search"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # Bayesian
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        objective,
        n_trials=100,
        timeout=3600,  # 1 hour max
        n_jobs=1
    )

    print(f"Best trial ROC AUC: {study.best_trial.value:.4f}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    # Retrain final model with best parameters on full training set
    best_model = GradientBoostingClassifier(**study.best_trial.params)
    best_model.fit(X_train, y_train)

    final_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("final_test_roc_auc", final_roc_auc)
    mlflow.sklearn.log_model(best_model, "best_model")
```

---

## Chapter 12: Model Evaluation & Validation

### 12.1 Beyond Accuracy — Choosing the Right Metrics

Accuracy is often a poor choice of evaluation metric for real-world ML problems. Consider a fraud detection model where 99.5% of transactions are legitimate. A model that always predicts "not fraud" achieves 99.5% accuracy — and is completely useless. Choosing the right metric is a modeling decision with real business consequences.

**Classification Metrics:**

```
Confusion Matrix:
                  Predicted Positive  Predicted Negative
Actual Positive       TP                  FN
Actual Negative       FP                  TN

Precision = TP / (TP + FP)     → Of all I labeled positive, how many actually are?
Recall    = TP / (TP + FN)     → Of all actual positives, how many did I find?
F1        = 2 * (P*R) / (P+R)  → Harmonic mean of precision and recall
ROC AUC   = Area under ROC curve → Probability that model ranks a positive higher than a negative
PR AUC    = Area under Precision-Recall curve → Better for imbalanced datasets
```

**When to use which metric:**
- Fraud detection (asymmetric cost of false negatives): prioritize **Recall** — missing fraud is very costly
- Spam filter (user prefers fewer false alarms): prioritize **Precision** — incorrectly marking ham as spam is very annoying
- Medical diagnosis (balanced concern): use **F1** or **ROC AUC**
- Imbalanced datasets: use **PR AUC** rather than ROC AUC

```python
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Full classification report
print(classification_report(y_test, y_pred,
                             target_names=["Retained", "Churned"]))

# ROC and PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax1)
ax1.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax1.set_title(f"ROC Curve (AUC={roc_auc_score(y_test, y_pred_proba):.3f})")

PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
ax2.set_title(f"Precision-Recall Curve (AP={average_precision_score(y_test, y_pred_proba):.3f})")

plt.tight_layout()
plt.savefig("evaluation_curves.png")
```

### 12.2 Business Metric Validation

Technical metrics (ROC AUC, F1) are proxies for business value. Before deploying, you must validate that improving the proxy metric actually improves the business outcome. This requires a **decision threshold analysis** — understanding the trade-off between precision and recall at different probability cutoffs.

```python
import numpy as np
import pandas as pd

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Business parameters
RETENTION_CAMPAIGN_COST = 50      # Cost to reach out to one at-risk customer
CUSTOMER_LIFETIME_VALUE = 800     # Revenue from retaining a churner

# Analyze profit at different thresholds
thresholds = np.arange(0.1, 0.9, 0.02)
results = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    tp = ((y_pred_thresh == 1) & (y_test == 1)).sum()  # True positives (retained churners)
    fp = ((y_pred_thresh == 1) & (y_test == 0)).sum()  # False positives (wasted campaigns)
    fn = ((y_pred_thresh == 0) & (y_test == 1)).sum()  # Missed churners

    profit = (tp * CUSTOMER_LIFETIME_VALUE) - ((tp + fp) * RETENTION_CAMPAIGN_COST)

    results.append({
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp + 1e-9),
        "recall": tp / (tp + fn + 1e-9),
        "profit": profit
    })

results_df = pd.DataFrame(results)
best_threshold = results_df.loc[results_df['profit'].idxmax(), 'threshold']
print(f"Optimal threshold for maximum profit: {best_threshold:.2f}")
print(f"Expected profit: ${results_df['profit'].max():,.0f}")
```

### 12.3 Model Comparison and Statistical Testing

When comparing two models, never choose based on a single number from a single test set. Use statistical tests to determine whether performance differences are meaningful.

```python
from scipy import stats
from sklearn.model_selection import cross_val_score
import numpy as np

# Compare two models using paired t-test on cross-validation scores
cv_scores_model_a = cross_val_score(model_a, X, y, cv=10, scoring='roc_auc')
cv_scores_model_b = cross_val_score(model_b, X, y, cv=10, scoring='roc_auc')

t_stat, p_value = stats.ttest_rel(cv_scores_model_a, cv_scores_model_b)

print(f"Model A: {cv_scores_model_a.mean():.4f} ± {cv_scores_model_a.std():.4f}")
print(f"Model B: {cv_scores_model_b.mean():.4f} ± {cv_scores_model_b.std():.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    better = "A" if cv_scores_model_a.mean() > cv_scores_model_b.mean() else "B"
    print(f"Model {better} is statistically significantly better (p < 0.05)")
else:
    print("No statistically significant difference — choose based on other criteria")
```

---

# PART IV — DEPLOYMENT

---

## Chapter 13: Model Registry & Versioning

### 13.1 The Model Registry Concept

A **Model Registry** is a centralized catalog that stores trained model artifacts and manages their lifecycle from experimentation to production. Think of it as a specialized version control system for models, analogous to an artifact registry (Docker Hub, npm registry) for software artifacts.

The registry serves as the single source of truth for "which model should be serving production traffic right now." Without it, teams resort to ad hoc file naming conventions (model_v3_final_REALLY_final.pkl), Slack messages, and institutional memory — all of which fail at scale.

### 13.2 MLflow Model Registry

MLflow's Model Registry provides lifecycle stage management with four stages: **None** (newly registered), **Staging** (under evaluation), **Production** (serving live traffic), and **Archived** (retired).

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model from a completed run
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="churn-prediction-model"
)
print(f"Model version: {registered_model.version}")

# Add a description
client.update_model_version(
    name="churn-prediction-model",
    version=registered_model.version,
    description="GBM model trained on 2024 Q1 data. ROC AUC: 0.923"
)

# Transition to Staging for testing
client.transition_model_version_stage(
    name="churn-prediction-model",
    version=registered_model.version,
    stage="Staging",
    archive_existing_versions=False
)

# After validation passes, promote to Production
client.transition_model_version_stage(
    name="churn-prediction-model",
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True  # Archive previous production version
)

# Load the current Production model (always gets the right version)
production_model = mlflow.sklearn.load_model(
    model_uri="models:/churn-prediction-model/Production"
)

# Load a specific version
specific_model = mlflow.sklearn.load_model(
    model_uri="models:/churn-prediction-model/3"
)

# List all versions
versions = client.get_registered_model("churn-prediction-model")
for mv in versions.latest_versions:
    print(f"  Version {mv.version}: {mv.current_stage}")
```

---

## Chapter 14: Model Serving Patterns

### 14.1 The Three Serving Patterns

The same trained model can be served in fundamentally different ways. Choosing the right pattern is a critical architectural decision that affects latency, throughput, cost, and complexity.

**Pattern 1: Batch Inference**

Batch inference runs predictions on a large dataset at a scheduled time. The results are stored and consumed downstream (in a database, a report, an email). There is no real-time latency requirement.

Use batch inference when: predictions don't need to be immediate; the dataset to score is large; you want to pre-compute predictions for all users nightly.

```
Batch Pattern:
Data Store ──► Feature Pipeline ──► Model ──► Prediction Store ──► Downstream App
(runs nightly)                              (database/S3)           (reads pre-computed)
```

**Pattern 2: Real-Time Online Inference**

A model is wrapped in an HTTP API endpoint. Client applications send a request with input features and receive a prediction response within milliseconds. This requires the model to be always-on and respond quickly.

Use online inference when: the user is waiting for the response; the input isn't known until request time; low latency (<100ms) is required.

**Pattern 3: Streaming Inference**

The model processes events from a message queue (Kafka, Pub/Sub) as they arrive. This is ideal for event-driven architectures where predictions need to be made on a continuous stream of events without polling.

Use streaming inference for: fraud detection on transactions as they happen; anomaly detection on IoT sensor streams; real-time recommendation on click events.

### 14.2 Batch Inference Implementation

```python
# batch_inference.py — Run predictions on a full dataset

import pandas as pd
import mlflow
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def run_batch_inference(
    input_path: str,
    output_path: str,
    model_name: str = "churn-prediction-model",
    model_stage: str = "Production"
):
    """
    Load production model and run inference on a batch dataset.
    Results are written to a Parquet file for downstream consumption.
    """
    logger.info(f"Starting batch inference at {datetime.now()}")

    # Load the production model
    model_uri = f"models:/{model_name}/{model_stage}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Loaded model: {model_uri}")

    # Load input data in chunks (for memory efficiency on large datasets)
    chunk_size = 10000
    predictions_list = []

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        customer_ids = chunk['customer_id']
        features = chunk.drop(columns=['customer_id'])

        # Predict
        churn_proba = model.predict_proba(features)[:, 1]

        chunk_predictions = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': churn_proba,
            'churn_predicted': (churn_proba >= 0.45).astype(int),
            'inference_timestamp': datetime.now(),
            'model_version': model_stage
        })

        predictions_list.append(chunk_predictions)
        logger.info(f"Processed chunk {i+1} ({len(chunk)} rows)")

    predictions_df = pd.concat(predictions_list, ignore_index=True)
    predictions_df.to_parquet(output_path, index=False)

    logger.info(f"Batch inference complete. {len(predictions_df)} predictions written to {output_path}")
    logger.info(f"Predicted churn rate: {predictions_df['churn_predicted'].mean():.3f}")

if __name__ == "__main__":
    run_batch_inference(
        input_path="data/scoring/customers_to_score.csv",
        output_path="data/predictions/churn_scores_2024_01_15.parquet"
    )
```

---

## Chapter 15: Model Deployment with FastAPI & BentoML

### 15.1 FastAPI — Building a Model Serving API

**FastAPI** is a modern Python web framework ideal for wrapping ML models in high-performance REST APIs. It automatically generates OpenAPI (Swagger) documentation and uses Python type hints for input validation.

```python
# serve/app.py — Production-ready FastAPI model server

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import mlflow
import pandas as pd
import numpy as np
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction",
    version="2.1.0"
)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total', 'Total predictions made', ['model_version', 'prediction']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 'Prediction latency in seconds'
)
REQUEST_ERRORS = Counter('request_errors_total', 'Total request errors')

# Load model at startup (not per-request!)
model = None
model_version = None

@app.on_event("startup")
async def load_model():
    global model, model_version
    try:
        model_uri = "models:/churn-prediction-model/Production"
        model = mlflow.sklearn.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("churn-prediction-model", stages=["Production"])
        model_version = versions[0].version if versions else "unknown"
        logger.info(f"Model loaded: version {model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Request/Response schemas
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=120, description="Customer age")
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly billing amount")
    total_purchases_30d: int = Field(..., ge=0)
    avg_session_duration_mins: float = Field(..., ge=0)
    days_since_last_login: int = Field(..., ge=0)
    support_tickets_90d: int = Field(..., ge=0)
    plan_type: str = Field(..., description="One of: free, basic, premium, enterprise")

    @validator('plan_type')
    def validate_plan_type(cls, v):
        valid_plans = {'free', 'basic', 'premium', 'enterprise'}
        if v not in valid_plans:
            raise ValueError(f"plan_type must be one of {valid_plans}")
        return v

class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    risk_tier: str  # "low", "medium", "high"
    model_version: str

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    inference_time_ms: float
    model_version: str

def features_to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to pandas DataFrame for model input."""
    plan_mapping = {'free': 0, 'basic': 1, 'premium': 2, 'enterprise': 3}
    data = {
        'age': [features.age],
        'tenure_months': [features.tenure_months],
        'monthly_charges': [features.monthly_charges],
        'total_purchases_30d': [features.total_purchases_30d],
        'avg_session_duration_mins': [features.avg_session_duration_mins],
        'days_since_last_login': [features.days_since_last_login],
        'support_tickets_90d': [features.support_tickets_90d],
        'plan_type_encoded': [plan_mapping.get(features.plan_type, 0)],
    }
    return pd.DataFrame(data)

def get_risk_tier(probability: float) -> str:
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "medium"
    else:
        return "high"

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerFeatures):
    """Predict churn probability for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features_df = features_to_dataframe(customer)
        churn_proba = float(model.predict_proba(features_df)[0, 1])
        latency = time.time() - start_time

        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(
            model_version=model_version,
            prediction="churn" if churn_proba >= 0.45 else "retain"
        ).inc()

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(churn_proba, 4),
            churn_predicted=churn_proba >= 0.45,
            risk_tier=get_risk_tier(churn_proba),
            model_version=model_version
        )
    except Exception as e:
        REQUEST_ERRORS.inc()
        logger.error(f"Prediction error for {customer.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict churn for a batch of customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Build feature matrix for all customers at once
        features_list = [features_to_dataframe(c) for c in request.customers]
        features_df = pd.concat(features_list, ignore_index=True)

        # Single model.predict_proba call for efficiency
        churn_probas = model.predict_proba(features_df)[:, 1]

        predictions = []
        for customer, proba in zip(request.customers, churn_probas):
            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=round(float(proba), 4),
                churn_predicted=float(proba) >= 0.45,
                risk_tier=get_risk_tier(float(proba)),
                model_version=model_version
            ))

        inference_time = (time.time() - start_time) * 1000

        return BatchResponse(
            predictions=predictions,
            inference_time_ms=round(inference_time, 2),
            model_version=model_version
        )
    except Exception as e:
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")

# Run: uvicorn serve.app:app --host 0.0.0.0 --port 8080 --workers 4
```

### 15.2 BentoML — ML-Native Serving Framework

BentoML is purpose-built for ML serving. Unlike FastAPI (a general web framework), BentoML understands ML concepts natively — it handles model loading, serialization, adaptive batching, and multi-model pipelines out of the box.

```python
# bento_service.py — Serving with BentoML

import bentoml
from bentoml.io import JSON, NumpyNdarray, PandasDataFrame
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd

# Save model to BentoML's model store
import mlflow
model = mlflow.sklearn.load_model("models:/churn-prediction-model/Production")
bento_model = bentoml.sklearn.save_model(
    "churn_gbm",
    model,
    signatures={
        "predict": {"batchable": True, "batch_dim": 0},
        "predict_proba": {"batchable": True, "batch_dim": 0}
    },
    metadata={"accuracy": 0.923, "training_dataset": "customers_2024_q1"}
)
print(f"Model saved: {bento_model.tag}")

# Define the service
svc = bentoml.Service("churn_prediction_service", runners=[
    bentoml.sklearn.get("churn_gbm:latest").to_runner()
])

class ChurnInput(BaseModel):
    customer_id: str
    age: int
    tenure_months: int
    monthly_charges: float
    total_purchases_30d: int

class ChurnOutput(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool

@svc.api(input=JSON(pydantic_model=ChurnInput),
         output=JSON(pydantic_model=ChurnOutput))
async def predict(input_data: ChurnInput) -> ChurnOutput:
    runner = svc.runners[0]
    features = np.array([[
        input_data.age,
        input_data.tenure_months,
        input_data.monthly_charges,
        input_data.total_purchases_30d,
    ]])
    proba = await runner.predict_proba.async_run(features)
    churn_proba = float(proba[0, 1])
    return ChurnOutput(
        customer_id=input_data.customer_id,
        churn_probability=round(churn_proba, 4),
        churn_predicted=churn_proba >= 0.45
    )
```

```bash
# Serve locally
bentoml serve bento_service:svc --reload

# Build a Bento (packaged service with all dependencies)
bentoml build

# Containerize the Bento
bentoml containerize churn_prediction_service:latest

# Run the container
docker run -p 3000:3000 churn_prediction_service:latest
```

---

## Chapter 16: Containerizing ML Models

### 16.1 Why Containerize ML Models

A trained model isn't just a .pkl file — it depends on a specific Python version, specific library versions (sklearn 1.3.0 not 1.2.0), sometimes specific system libraries (CUDA for GPU inference), and a specific way to load and run it. Without containerization, "it works on my machine" becomes the defining failure mode of ML deployment.

Docker containers solve this by packaging the model, code, dependencies, and runtime together into an immutable artifact that runs identically everywhere — a laptop, a CI server, or a Kubernetes cluster.

### 16.2 Writing a Production Dockerfile for ML

```dockerfile
# Dockerfile — Multi-stage build for production ML service

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime image (smaller)
FROM python:3.11-slim as runtime

WORKDIR /app

# Create non-root user (security best practice)
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY serve/ serve/
COPY models/ models/

# Set Python path
ENV PATH=/home/mluser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.pkl
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER mluser

# Expose port
EXPOSE ${PORT}

# Start server
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080",
     "--workers", "1", "--log-level", "info"]
```

```
# requirements.txt — Pin ALL versions for reproducibility
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
mlflow==2.9.2
prometheus-client==0.19.0
pydantic==2.5.2
```

```bash
# Build
docker build -t churn-prediction:v2.1.0 .

# Test locally
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  churn-prediction:v2.1.0

# Push to registry
docker tag churn-prediction:v2.1.0 us-central1-docker.pkg.dev/my-project/ml-models/churn-prediction:v2.1.0
docker push us-central1-docker.pkg.dev/my-project/ml-models/churn-prediction:v2.1.0

# Run on Kubernetes
kubectl apply -f k8s/deployment.yaml
```

```yaml
# k8s/deployment.yaml — Kubernetes deployment for ML service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction
  labels:
    app: churn-prediction
    version: v2.1.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
  template:
    metadata:
      labels:
        app: churn-prediction
        version: v2.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8080"
    spec:
      containers:
      - name: churn-prediction
        image: us-central1-docker.pkg.dev/my-project/ml-models/churn-prediction:v2.1.0
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: tracking_uri
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 20
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: churn-prediction-svc
spec:
  selector:
    app: churn-prediction
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

---

# PART V — AUTOMATION & ORCHESTRATION

---

## Chapter 17: CI/CD for Machine Learning

### 17.1 What CI/CD Means in ML Context

In traditional software, CI/CD automates the process of integrating code changes and deploying them to production. In ML, CI/CD must automate a much richer set of activities:

**Continuous Integration (CI)** in ML means: on every code change, automatically run data validation tests, feature engineering tests, model training (or a fast smoke-test training), model evaluation against quality thresholds, and integration tests for the serving code.

**Continuous Delivery (CD)** in ML means: if CI passes, automatically promote the model artifact through staging and production environments, perform canary deployments, and monitor the new model version before routing full traffic to it.

**Continuous Training (CT)** is an additional concept unique to ML: automatically retrain the model when new data arrives or when model performance degrades below a threshold. This has no equivalent in traditional software CI/CD.

### 17.2 GitHub Actions for ML CI/CD

```yaml
# .github/workflows/ml-ci-cd.yml

name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # Nightly retraining trigger
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  MODEL_NAME: 'churn-prediction-model'
  IMAGE_NAME: 'churn-prediction'
  GCP_REGION: 'us-central1'

jobs:
  # ─────────────────────────────────────────────
  # Job 1: Data Validation
  # ─────────────────────────────────────────────
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements/test.txt

      - name: Pull data with DVC
        run: |
          pip install dvc[s3]
          dvc pull data/processed/
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Run data validation
        run: |
          python -m pytest tests/test_data_validation.py -v
          python scripts/validate_schema.py --data=data/processed/train.csv

  # ─────────────────────────────────────────────
  # Job 2: Code Tests
  # ─────────────────────────────────────────────
  code-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements/test.txt

      - name: Lint with flake8
        run: flake8 src/ tests/ --max-line-length=100

      - name: Type check with mypy
        run: mypy src/

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Test feature engineering functions
        run: pytest tests/test_features.py -v

      - name: Test preprocessing pipeline
        run: pytest tests/test_pipeline.py -v

  # ─────────────────────────────────────────────
  # Job 3: Model Training
  # ─────────────────────────────────────────────
  train-model:
    needs: [data-validation, code-tests]
    runs-on: ubuntu-latest
    outputs:
      run_id: ${{ steps.train.outputs.run_id }}
      roc_auc: ${{ steps.train.outputs.roc_auc }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull data
        run: dvc pull

      - name: Train model
        id: train
        run: |
          python src/train.py --experiment-name "ci-cd-${{ github.sha }}"
          # Read outputs written by train.py
          echo "run_id=$(cat artifacts/run_id.txt)" >> $GITHUB_OUTPUT
          echo "roc_auc=$(cat artifacts/roc_auc.txt)" >> $GITHUB_OUTPUT
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: models/

  # ─────────────────────────────────────────────
  # Job 4: Model Evaluation Gate
  # ─────────────────────────────────────────────
  model-evaluation-gate:
    needs: train-model
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check model quality threshold
        run: |
          ROC_AUC=${{ needs.train-model.outputs.roc_auc }}
          THRESHOLD=0.85
          if (( $(echo "$ROC_AUC < $THRESHOLD" | bc -l) )); then
            echo "FAIL: ROC AUC $ROC_AUC is below threshold $THRESHOLD"
            exit 1
          fi
          echo "PASS: ROC AUC $ROC_AUC meets threshold $THRESHOLD"

      - name: Compare with production model
        run: |
          python scripts/compare_with_production.py \
            --new-run-id ${{ needs.train-model.outputs.run_id }} \
            --model-name ${{ env.MODEL_NAME }}
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

  # ─────────────────────────────────────────────
  # Job 5: Build and Push Docker Image
  # ─────────────────────────────────────────────
  build-image:
    needs: model-evaluation-gate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifacts
        uses: actions/download-artifact@v3
        with:
          name: model-artifacts
          path: models/

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Configure Docker for GCP
        run: gcloud auth configure-docker ${{ env.GCP_REGION }}-docker.pkg.dev

      - name: Build and push Docker image
        run: |
          IMAGE_TAG="${{ env.GCP_REGION }}-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/ml-models/${{ env.IMAGE_NAME }}:${{ github.sha }}"
          docker build -t $IMAGE_TAG .
          docker push $IMAGE_TAG

  # ─────────────────────────────────────────────
  # Job 6: Deploy to Staging
  # ─────────────────────────────────────────────
  deploy-staging:
    needs: build-image
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to Cloud Run (staging)
        run: |
          gcloud run deploy ${{ env.IMAGE_NAME }}-staging \
            --image="${{ env.GCP_REGION }}-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/ml-models/${{ env.IMAGE_NAME }}:${{ github.sha }}" \
            --region=${{ env.GCP_REGION }} \
            --platform=managed \
            --allow-unauthenticated \
            --no-traffic

      - name: Run smoke tests against staging
        run: python tests/smoke_tests.py --env staging

  # ─────────────────────────────────────────────
  # Job 7: Deploy to Production
  # ─────────────────────────────────────────────
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # Requires manual approval in GitHub
    steps:
      - name: Canary deploy (10% traffic)
        run: |
          gcloud run deploy ${{ env.IMAGE_NAME }} \
            --image="${{ env.GCP_REGION }}-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/ml-models/${{ env.IMAGE_NAME }}:${{ github.sha }}" \
            --region=${{ env.GCP_REGION }} \
            --platform=managed \
            --no-traffic

          gcloud run services update-traffic ${{ env.IMAGE_NAME }} \
            --region=${{ env.GCP_REGION }} \
            --to-latest=10

      - name: Monitor canary for 10 minutes
        run: sleep 600

      - name: Promote to 100% traffic
        run: |
          gcloud run services update-traffic ${{ env.IMAGE_NAME }} \
            --region=${{ env.GCP_REGION }} \
            --to-latest=100

      - name: Register model in MLflow Registry as Production
        run: |
          python scripts/promote_to_production.py \
            --run-id ${{ needs.train-model.outputs.run_id }} \
            --model-name ${{ env.MODEL_NAME }}
```

---

## Chapter 18: ML Pipeline Orchestration

### 18.1 Why Orchestration is Necessary

An ML pipeline is a DAG of steps: ingest data → validate → compute features → train → evaluate → deploy. Running these steps manually works in development. In production, you need them to run automatically, on a schedule or triggered by events, with proper dependency management, retries on failure, and visibility into what ran, when, and with what results.

**Apache Airflow** is the most widely deployed orchestrator for data and ML pipelines. It defines pipelines as Python code (DAGs), provides a rich UI for monitoring, and has connectors for every major data system.

### 18.2 Airflow ML Pipeline

```python
# dags/ml_training_pipeline.py — Full ML pipeline DAG

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.task_group import TaskGroup
import logging

logger = logging.getLogger(__name__)

# Default arguments applied to all tasks
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['mlops-alerts@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

def ingest_data(**context):
    """Extract latest data from data warehouse."""
    from src.data.ingest import run_ingestion
    execution_date = context['execution_date']
    output_path = f"data/raw/customers_{execution_date.strftime('%Y%m%d')}.csv"
    rows_ingested = run_ingestion(
        start_date=execution_date - timedelta(days=30),
        end_date=execution_date,
        output_path=output_path
    )
    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='raw_data_path', value=output_path)
    context['ti'].xcom_push(key='rows_ingested', value=rows_ingested)
    logger.info(f"Ingested {rows_ingested} rows to {output_path}")

def validate_data(**context):
    """Run Great Expectations validation."""
    from src.data.validate import run_validation
    raw_data_path = context['ti'].xcom_pull(key='raw_data_path')
    result = run_validation(raw_data_path, suite_name="production_suite")
    if not result.success:
        raise ValueError(f"Data validation failed: {result.to_json_dict()}")
    logger.info("Data validation passed")

def compute_features(**context):
    """Compute features and write to feature store."""
    from src.features.pipeline import run_feature_pipeline
    raw_data_path = context['ti'].xcom_pull(key='raw_data_path')
    features_path = run_feature_pipeline(raw_data_path)
    context['ti'].xcom_push(key='features_path', value=features_path)

def train_model(**context):
    """Train model and log to MLflow."""
    from src.train import run_training
    import mlflow

    features_path = context['ti'].xcom_pull(key='features_path')
    run_id, metrics = run_training(
        features_path=features_path,
        experiment_name="production-weekly-training"
    )
    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='roc_auc', value=metrics['roc_auc'])
    logger.info(f"Training complete. Run ID: {run_id}, ROC AUC: {metrics['roc_auc']:.4f}")

def evaluate_model_quality(**context):
    """Check if model meets quality threshold. Branch to deploy or reject."""
    roc_auc = context['ti'].xcom_pull(key='roc_auc')
    THRESHOLD = 0.85
    if roc_auc >= THRESHOLD:
        logger.info(f"Model quality PASS: ROC AUC {roc_auc:.4f} >= {THRESHOLD}")
        return 'deploy_to_staging'
    else:
        logger.warning(f"Model quality FAIL: ROC AUC {roc_auc:.4f} < {THRESHOLD}")
        return 'reject_model'

def deploy_to_staging(**context):
    from src.deploy import deploy_to_cloud_run
    run_id = context['ti'].xcom_pull(key='run_id')
    deploy_to_cloud_run(run_id=run_id, environment='staging')

def run_integration_tests(**context):
    from src.test import run_smoke_tests
    result = run_smoke_tests(environment='staging')
    if not result.passed:
        raise ValueError(f"Integration tests failed: {result.failures}")

def promote_to_production(**context):
    from src.deploy import deploy_to_cloud_run
    from src.registry import promote_model
    run_id = context['ti'].xcom_pull(key='run_id')
    deploy_to_cloud_run(run_id=run_id, environment='production')
    promote_model(run_id=run_id, model_name="churn-prediction-model", stage="Production")

def reject_model(**context):
    roc_auc = context['ti'].xcom_pull(key='roc_auc')
    logger.warning(f"Model rejected. ROC AUC: {roc_auc:.4f} below threshold 0.85")

# ─── Define the DAG ───────────────────────────────────────

with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='Weekly ML model training and deployment pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'production'],
    max_active_runs=1
) as dag:

    task_ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
    )

    task_validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )

    task_features = PythonOperator(
        task_id='compute_features',
        python_callable=compute_features,
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    task_evaluate = BranchPythonOperator(
        task_id='evaluate_model_quality',
        python_callable=evaluate_model_quality,
    )

    task_deploy_staging = PythonOperator(
        task_id='deploy_to_staging',
        python_callable=deploy_to_staging,
    )

    task_integration_tests = PythonOperator(
        task_id='run_integration_tests',
        python_callable=run_integration_tests,
    )

    task_promote = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_to_production,
    )

    task_reject = PythonOperator(
        task_id='reject_model',
        python_callable=reject_model,
    )

    task_notify_success = SlackWebhookOperator(
        task_id='notify_success',
        http_conn_id='slack_webhook',
        message=":white_check_mark: New churn model deployed to production! Run: {{ ti.xcom_pull(key='run_id') }}",
    )

    # Define task dependencies (the DAG structure)
    task_ingest >> task_validate >> task_features >> task_train >> task_evaluate
    task_evaluate >> task_deploy_staging >> task_integration_tests >> task_promote >> task_notify_success
    task_evaluate >> task_reject
```

---

# PART VI — MONITORING & RELIABILITY

---

## Chapter 20: ML Monitoring — Why Models Degrade

### 20.1 The Fundamental Problem

A trained model is a snapshot of a relationship between inputs and outputs at a point in time. The world, however, is not static. User behaviors change, economies shift, regulations are enacted, new products are launched, and natural disasters disrupt patterns. The model, frozen at training time, cannot adapt to these changes on its own.

This degradation is inevitable and cannot be prevented — it can only be detected and addressed through continuous monitoring and retraining. The goal of ML monitoring is to answer one question continuously: "Is this model still delivering the value we expect?"

There are two distinct types of degradation to monitor: **data drift** (the inputs to the model have changed) and **concept drift** (the relationship between inputs and output has changed).

### 20.2 Understanding Data Drift vs. Concept Drift

**Data Drift** (also called feature drift or covariate shift) occurs when the statistical distribution of input features changes over time, without the relationship between inputs and output necessarily changing.

> **Example**: Your fraud detection model was trained when most users were in the US. Your company expanded to Southeast Asia. The distribution of IP geolocation features shifted dramatically. The model may still work correctly on US users but performs poorly on the new user base — not because fraud patterns changed, but because the input distribution changed.

**Concept Drift** occurs when the statistical relationship between inputs and output changes. The fundamental pattern the model learned is no longer valid.

> **Example**: Before COVID, "large purchase + international location + new account" was a strong fraud signal. During COVID, international travel stopped but online international purchases increased dramatically. The same input combination now has a different meaning (lower fraud probability). The model's learned concept is outdated.

```
┌────────────────────────────────────────────────────────────┐
│              TYPES OF DISTRIBUTION SHIFT                    │
│                                                            │
│  P(X) changes          = Data Drift / Covariate Shift      │
│  P(Y|X) changes        = Concept Drift                     │
│  P(Y) changes          = Label Shift / Prior Probability   │
│  All of the above      = Dataset Shift                     │
│                                                            │
│  Training data: P_train(X,Y)                               │
│  Serving data:  P_serve(X,Y)                 
