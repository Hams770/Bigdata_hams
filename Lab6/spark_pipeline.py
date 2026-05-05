from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (StringIndexer, OneHotEncoder,
                                 VectorAssembler, StandardScaler)
from pyspark.ml.classification import (LogisticRegression,
                                        RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import broadcast
import time

# ─────────────────────────────────────────────
# SparkSession
# ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Lab9A-Foundations") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)

# ─────────────────────────────────────────────
# Task 2.1 — Dataset Preparation
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TASK 2.1 — Dataset Preparation")
print("="*60)

column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# Load CSV from HDFS (adjust path if needed)
df = spark.read.csv(
    "/user/student/lab9/adult.csv",
    header=False,
    inferSchema=True
).toDF(*column_names)

# Trim whitespace from string columns
for c in df.columns:
    if dict(df.dtypes)[c] == "string":
        df = df.withColumn(c, F.trim(F.col(c)))

# Print schema
print("\nSchema:")
df.printSchema()

# Show first 5 rows
print("First 5 rows:")
df.show(5, truncate=False)

# Total record count
total = df.count()
print(f"Total records: {total:,}")

# Create binary label: 1 if income ">50K", else 0
df = df.withColumn("label", (F.col("income") == ">50K").cast("int"))

# Class distribution
print("\nClass distribution:")
dist = df.groupBy("label").count().withColumn(
    "percentage", F.round(F.col("count") / total * 100, 2)
)
dist.show()
# Dataset is typically imbalanced (~76% ≤50K, ~24% >50K)

# ─────────────────────────────────────────────
# Task 2.2 — MLlib Pipeline
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TASK 2.2 — MLlib Pipeline")
print("="*60)

categorical_cols = ["workclass", "education", "marital_status",
                    "occupation", "relationship", "race",
                    "sex", "native_country"]
numeric_cols = ["age", "fnlwgt", "education_num", "capital_gain",
                "capital_loss", "hours_per_week"]

# Stage 1: StringIndexer per categorical column
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx",
                          handleInvalid="keep")
            for c in categorical_cols]

# Stage 2: OneHotEncoder
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe")
            for c in categorical_cols]

# Stage 3: VectorAssembler
assembler = VectorAssembler(
    inputCols=[c + "_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="raw_features")

# Stage 4: StandardScaler
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=False, withStd=True)

# Stage 5: Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label",
                        maxIter=20, regParam=0.01)

pipeline_lr = Pipeline(stages=indexers + encoders + [assembler, scaler, lr])

# Train/test split 80/20
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train size: {train_df.count():,} | Test size: {test_df.count():,}")

# Train Logistic Regression pipeline
print("\nTraining Logistic Regression pipeline...")
t0 = time.time()
model_lr = pipeline_lr.fit(train_df)
lr_train_time = time.time() - t0
print(f"LR Training time: {lr_train_time:.3f}s")

# Evaluate
evaluator_roc = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderROC")
evaluator_pr = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderPR")

preds_lr = model_lr.transform(test_df)
auc_roc_lr = evaluator_roc.evaluate(preds_lr)
auc_pr_lr = evaluator_pr.evaluate(preds_lr)
print(f"LR AUC-ROC: {auc_roc_lr:.4f}")
print(f"LR AUC-PR:  {auc_pr_lr:.4f}")

# Replace with Random Forest
print("\nTraining Random Forest pipeline...")
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=100, seed=42)
pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])

t0 = time.time()
model_rf = pipeline_rf.fit(train_df)
rf_train_time = time.time() - t0
print(f"RF Training time: {rf_train_time:.3f}s")

preds_rf = model_rf.transform(test_df)
auc_roc_rf = evaluator_roc.evaluate(preds_rf)
auc_pr_rf = evaluator_pr.evaluate(preds_rf)
print(f"RF AUC-ROC: {auc_roc_rf:.4f}")
print(f"RF AUC-PR:  {auc_pr_rf:.4f}")

print(f"\nComparison: LR={auc_roc_lr:.4f}  RF={auc_roc_rf:.4f}")
better = "Random Forest" if auc_roc_rf > auc_roc_lr else "Logistic Regression"
print(f"Better model: {better}")

# Print ordered stages
print("\nPipeline stages (LR pipeline):")
for i, stage in enumerate(model_lr.stages):
    print(f"  Stage {i:2d}: {type(stage).__name__}")
print(f"Total stages: {len(model_lr.stages)}")

# Execution plan
print("\nExecution plan for train_df:")
train_df.explain(True)

# ─────────────────────────────────────────────
# Task 2.3 — Partition Scaling Experiment
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TASK 2.3 — Partition Scaling Experiment")
print("="*60)

print(f"train_df partitions before repartition: {train_df.rdd.getNumPartitions()}")

partition_counts = [1, 2, 4, 8, 16]
results = []

for n_parts in partition_counts:
    df_repartitioned = train_df.repartition(n_parts)
    print(f"  After repartition({n_parts}): {df_repartitioned.rdd.getNumPartitions()} partitions")
    t0 = time.time()
    model = pipeline_lr.fit(df_repartitioned)
    elapsed = time.time() - t0
    results.append((n_parts, round(elapsed, 3)))
    print(f"Partitions: {n_parts:>3} | Training time: {elapsed:.3f}s")

print("\nSummary table:")
print(f"{'Partitions':>12} {'Training Time (s)':>18}")
print("-" * 32)
for n_parts, t in results:
    print(f"{n_parts:>12} {t:>18.3f}")

best = min(results, key=lambda x: x[1])
print(f"\nOptimal partition count: {best[0]} (time: {best[1]:.3f}s)")

# ─────────────────────────────────────────────
# Task 3.2 — Broadcast vs. Exchange in Spark
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TASK 3.2 — Broadcast vs. Shuffle Join")
print("="*60)

# Small lookup table
lookup = spark.createDataFrame(
    [(i, f"category_{i}") for i in range(100)],
    ["id", "category"])

# Large table derived from training data
large = train_df.select(
    (F.col("age") % 100).alias("id"),
    F.col("label"))

# Without broadcast hint
t0 = time.time()
_ = large.join(lookup, on="id").count()
t_no_bc = time.time() - t0

# With broadcast hint
t0 = time.time()
_ = large.join(broadcast(lookup), on="id").count()
t_bc = time.time() - t0

print(f"Without broadcast: {t_no_bc:.3f}s")
print(f"With broadcast:    {t_bc:.3f}s")
print(f"Speedup:           {t_no_bc / t_bc:.2f}x")

print("\nExecution plan WITHOUT broadcast hint:")
large.join(lookup, on="id").explain()

print("\nExecution plan WITH broadcast hint:")
large.join(broadcast(lookup), on="id").explain()

spark.stop()