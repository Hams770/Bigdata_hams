# CS4074 Lab 5 -- NYC Taxi Analytics Pipeline
# Student Name : Hams Aljohani
# Student ID   : S20106833

import os, time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# ==============================================================================
# Part 3: Create the SparkSession
# ==============================================================================
spark = SparkSession.builder \
    .appName('CS4074_Lab5_NYC_Taxi') \
    .master('local[*]') \
    .config('spark.executor.memory', '4g') \
    .config('spark.driver.memory', '2g') \
    .config('spark.sql.shuffle.partitions', '50') \
    .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
    .config('spark.hadoop.fs.defaultFS', 'hdfs://localhost:9000') \
    .getOrCreate()

print(f"Spark Version : {spark.version}")
print(f"App Name      : {spark.sparkContext.appName}")
print(f"Master        : {spark.sparkContext.master}")
print("Open Spark UI : http://localhost:4040")

# ==============================================================================
# Part 4: Data Ingestion from HDFS
# ==============================================================================
HDFS_BASE = 'hdfs://localhost:9000/cs4074/lab5'

# --- 4.1 Read Parquet trip data (schema is embedded) ---
df_trips = spark.read.parquet(f'{HDFS_BASE}/data/yellow_tripdata_2024-01.parquet')

df_trips.printSchema()
print(f"Total rows  : {df_trips.count():,}")
print(f"Partitions  : {df_trips.rdd.getNumPartitions()}")
df_trips.show(5, truncate=False)

# --- 4.2 Read zone lookup CSV with explicit schema ---
zone_schema = StructType([
    StructField('LocationID',   IntegerType(), True),
    StructField('Borough',      StringType(),  True),
    StructField('Zone',         StringType(),  True),
    StructField('service_zone', StringType(),  True),
])

df_zones = spark.read \
    .option('header', 'true') \
    .schema(zone_schema) \
    .csv(f'{HDFS_BASE}/data/taxi_zone_lookup.csv')

df_zones.printSchema()
df_zones.show(10)

# --- 4.3 Explore null values ---
null_counts = df_trips.select([
    sum(col(c).isNull().cast('int')).alias(c)
    for c in df_trips.columns
])
null_counts.show()

# ==============================================================================
# Part 5: DataFrame Operations
# ==============================================================================

# --- 5.1 Filtering -- Narrow ---
df_valid = df_trips.filter(
    (col('fare_amount')   > 0) &
    (col('trip_distance') > 0) &
    (col('total_amount')  > 0) &
    (col('passenger_count') > 0)
)

original_count = df_trips.count()
valid_count    = df_valid.count()
removed        = original_count - valid_count

print(f"Original rows : {original_count:,}")
print(f"Valid rows    : {valid_count:,}")
print(f"Removed rows  : {removed:,} ({removed / original_count * 100:.1f}%)")

# --- 5.2 Adding computed columns -- Narrow ---
df_enriched = df_valid \
    .withColumn('tip_pct',
        round(col('tip_amount') / col('fare_amount') * 100, 2)) \
    .withColumn('trip_category',
        when(col('trip_distance') < 1,  'Short')
        .when(col('trip_distance') < 5,  'Medium')
        .when(col('trip_distance') < 15, 'Long')
        .otherwise('Very Long')) \
    .withColumn('pickup_hour', hour(col('tpep_pickup_datetime'))) \
    .withColumn('is_weekend',
        dayofweek(col('tpep_pickup_datetime')).isin([1, 7])) \
    .withColumn('is_airport_trip',
        (col('PULocationID').isin(132, 138)) |
        (col('DOLocationID').isin(132, 138)))

df_enriched.select(
    'trip_distance', 'trip_category', 'fare_amount',
    'tip_pct', 'pickup_hour', 'is_weekend', 'is_airport_trip'
).show(10)

# --- 5.3 Aggregations with groupBy -- Wide (hash partitioning shuffle) ---

# Revenue analysis by trip category
df_enriched.groupBy('trip_category').agg(
    count('*').alias('num_trips'),
    round(sum('total_amount'), 2).alias('total_revenue'),
    round(avg('fare_amount'),  2).alias('avg_fare'),
    round(avg('tip_pct'),      2).alias('avg_tip_pct'),
    round(avg('trip_distance'),2).alias('avg_distance'),
).orderBy(desc('total_revenue')).show()

# Peak hour analysis
df_enriched.groupBy('pickup_hour').agg(
    count('*').alias('num_trips'),
    round(avg('fare_amount'), 2).alias('avg_fare'),
).orderBy('pickup_hour').show(24)

# --- 5.4 Broadcast Join with Zone Lookup -- Narrow (no shuffle for zones) ---
pickup_zones = df_zones \
    .withColumnRenamed('LocationID', 'PULocationID') \
    .withColumnRenamed('Borough',    'pickup_borough') \
    .withColumnRenamed('Zone',       'pickup_zone') \
    .select('PULocationID', 'pickup_borough', 'pickup_zone')

df_with_zones = df_enriched.join(
    broadcast(pickup_zones),
    on='PULocationID',
    how='left'
)

# Verify: no Exchange node for the zones table
df_with_zones.explain()

# Revenue by borough
df_with_zones.groupBy('pickup_borough').agg(
    count('*').alias('num_trips'),
    round(sum('total_amount'), 2).alias('total_revenue'),
    round(avg('tip_pct'),      2).alias('avg_tip_pct'),
).orderBy(desc('total_revenue')).show()

# --- 5.5 Handling Null Values -- Narrow ---
null_borough = df_with_zones.filter(col('pickup_borough').isNull()).count()
print(f"Trips with no matching zone: {null_borough:,}")

df_clean = df_with_zones.fillna({
    'pickup_borough': 'Unknown',
    'pickup_zone':    'Unknown',
    'tip_pct':        0.0,
})

# --- 5.6 Sorting and Pivot Table -- Wide (global sort / shuffle + aggregation) ---

# Top 10 most expensive trips
df_clean.select(
    'tpep_pickup_datetime', 'pickup_borough', 'trip_distance',
    'fare_amount', 'tip_amount', 'total_amount'
).orderBy(desc('total_amount')).show(10, truncate=False)

# Pivot: average fare by borough x trip_category
df_clean.groupBy('pickup_borough') \
    .pivot('trip_category', ['Short', 'Medium', 'Long', 'Very Long']) \
    .agg(round(avg('fare_amount'), 2)) \
    .orderBy('pickup_borough') \
    .show()

# ==============================================================================
# Part 6: Spark SQL & Window Functions
# ==============================================================================
df_clean.createOrReplaceTempView('trips')
df_zones.createOrReplaceTempView('zones')
print("Views registered: trips, zones")

# Revenue breakdown by borough and trip category
spark.sql("""
    SELECT
        pickup_borough,
        trip_category,
        COUNT(*) AS num_trips,
        ROUND(SUM(total_amount), 2) AS total_revenue,
        ROUND(AVG(fare_amount),  2) AS avg_fare,
        ROUND(AVG(tip_pct),      2) AS avg_tip_pct
    FROM trips
    WHERE pickup_borough != 'Unknown'
      AND fare_amount > 0
    GROUP BY pickup_borough, trip_category
    ORDER BY pickup_borough, total_revenue DESC
""").show(30)

# Top 10 busiest hours
spark.sql("""
    SELECT
        pickup_hour,
        COUNT(*) AS num_trips,
        ROUND(AVG(fare_amount), 2) AS avg_fare
    FROM trips
    GROUP BY pickup_hour
    ORDER BY num_trips DESC
    LIMIT 10
""").show()

# Window functions: rank top 5 most expensive trips in each borough
ranked = spark.sql("""
    SELECT
        pickup_borough,
        pickup_zone,
        trip_distance,
        fare_amount,
        total_amount,
        RANK() OVER (
            PARTITION BY pickup_borough
            ORDER BY total_amount DESC
        ) AS rank_in_borough,
        ROUND(AVG(fare_amount) OVER (
            PARTITION BY pickup_borough
        ), 2) AS borough_avg_fare
    FROM trips
    WHERE pickup_borough NOT IN ('Unknown', 'EWR')
      AND fare_amount > 0
""")

ranked.filter(col('rank_in_borough') <= 5) \
    .orderBy('pickup_borough', 'rank_in_borough') \
    .show(30, truncate=False)

# Task 6.1 -- Same plan proof: DataFrame API vs SQL
print("=== DataFrame API Plan ===")
df_clean.filter(col('fare_amount') > 5) \
    .groupBy('pickup_borough') \
    .agg(sum('total_amount').alias('total')) \
    .explain()

print("=== SQL Plan ===")
spark.sql("""
    SELECT pickup_borough, SUM(total_amount) AS total
    FROM trips WHERE fare_amount > 5
    GROUP BY pickup_borough
""").explain()

# ==============================================================================
# Part 7: User-Defined Functions (UDF Benchmarks)
# ==============================================================================
from pyspark.sql.functions import udf
import pandas as pd
from pyspark.sql.functions import pandas_udf

# --- 7.1 Python UDF (Slowest) ---
def classify_fare(fare):
    if fare is None: return 'Unknown'
    elif fare < 5:   return 'Low'
    elif fare < 15:  return 'Standard'
    elif fare < 30:  return 'High'
    else:            return 'Premium'

classify_fare_udf = udf(classify_fare, StringType())

start = time.time()
df_clean.withColumn('fare_tier_udf', classify_fare_udf(col('fare_amount'))).count()
udf_time = time.time() - start
print(f"Python UDF time  : {udf_time:.2f}s")

# --- 7.2 Built-in when().otherwise() (Fastest) ---
start = time.time()
df_clean.withColumn('fare_tier_builtin',
    when(col('fare_amount') < 5,  'Low')
    .when(col('fare_amount') < 15, 'Standard')
    .when(col('fare_amount') < 30, 'High')
    .otherwise('Premium')
).count()
builtin_time = time.time() - start
print(f"Built-in time    : {builtin_time:.2f}s")

# --- 7.3 Pandas UDF (Vectorised) ---
@pandas_udf(StringType())
def classify_fare_fast(fares: pd.Series) -> pd.Series:
    return fares.apply(
        lambda f: 'Unknown'  if pd.isna(f) else
                  'Low'      if f < 5      else
                  'Standard' if f < 15     else
                  'High'     if f < 30     else 'Premium'
    )

start = time.time()
df_clean.withColumn('fare_tier_pandas', classify_fare_fast(col('fare_amount'))).count()
pandas_time = time.time() - start
print(f"Pandas UDF time  : {pandas_time:.2f}s")

# --- 7.4 Benchmark Report ---
print(f"\n=== Benchmark Results ===")
print(f"Python UDF  : {udf_time:.2f}s")
print(f"Pandas UDF  : {pandas_time:.2f}s")
print(f"Built-in    : {builtin_time:.2f}s")
print(f"Speedup Pandas vs Python  : {udf_time / pandas_time:.1f}x")
print(f"Speedup Built-in vs Python: {udf_time / builtin_time:.1f}x")

# ==============================================================================
# Part 8: Partitioning in Practice
# ==============================================================================

# Current state
print(f"df_clean partitions : {df_clean.rdd.getNumPartitions()}")
print(f"Default parallelism : {spark.sparkContext.defaultParallelism}")

# Row distribution across partitions
df_clean.withColumn('partition_id', spark_partition_id()) \
    .groupBy('partition_id') \
    .count() \
    .orderBy('partition_id') \
    .show(60)

# repartition() -- full shuffle, can increase OR decrease
df_by_borough = df_clean.repartition(50, 'pickup_borough')
print(f"After repartition(50, borough): {df_by_borough.rdd.getNumPartitions()}")

# coalesce() -- no shuffle, can only decrease
df_small = df_clean.coalesce(4)
print(f"After coalesce(4)             : {df_small.rdd.getNumPartitions()}")

# Verify coalesce explain (should NOT show Exchange)
print("=== coalesce(4) Explain Plan ===")
df_small.explain()

# ==============================================================================
# Part 9: Caching & Persistence
# ==============================================================================
from pyspark import StorageLevel

df_clean.cache()   # = persist(MEMORY_ONLY)

# First access -- populates cache
start = time.time()
df_clean.count()
cache_build_time = time.time() - start
print(f"First access (builds cache) : {cache_build_time:.2f}s")
print(f"Is cached                   : {df_clean.is_cached}")

# Second access -- reads from cache
start = time.time()
df_clean.count()
cache_read_time = time.time() - start
print(f"Second access (from cache)  : {cache_read_time:.2f}s")
print(f"Speedup                     : {cache_build_time / cache_read_time:.1f}x faster")

input("Open http://localhost:4040 Storage tab, then press Enter...")

# Release cache
df_clean.unpersist()
print(f"After unpersist: {df_clean.is_cached}")

# ==============================================================================
# Part 10: Writing Results to HDFS
# ==============================================================================
borough_summary = df_clean.groupBy('pickup_borough', 'trip_category').agg(
    count('*').alias('num_trips'),
    round(sum('total_amount'), 2).alias('total_revenue'),
    round(avg('fare_amount'),  2).alias('avg_fare'),
    round(avg('tip_pct'),      2).alias('avg_tip_pct'),
    round(avg('trip_distance'),2).alias('avg_distance'),
).orderBy('pickup_borough', desc('total_revenue'))

# Write 1: Parquet with partition-aware subdirectories
borough_summary.write \
    .mode('overwrite') \
    .partitionBy('pickup_borough') \
    .parquet(f'{HDFS_BASE}/output/borough_summary_parquet')

# Write 2: Single CSV file (coalesce before writing)
borough_summary.coalesce(1) \
    .write \
    .mode('overwrite') \
    .option('header', 'true') \
    .csv(f'{HDFS_BASE}/output/borough_summary_csv')

print("Write completed!")

# ==============================================================================
# Part 11: Reading Execution Plans
# ==============================================================================
print("=== FULL PIPELINE PLAN ===")
borough_summary.explain(True)

print("=== FILTER + GROUPBY PLAN (High Fare Filter) ===")
df_clean.filter(col('fare_amount') > 3000) \
    .groupBy('pickup_borough') \
    .agg(sum('total_amount').alias('total')) \
    .explain()

# ==============================================================================
# Deep Work: Complete End-to-End Pipeline
# ==============================================================================
spark_dw = SparkSession.builder \
    .appName('CS4074_Lab5_DeepWork') \
    .master('local[4]') \
    .config('spark.executor.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '8') \
    .config('spark.hadoop.fs.defaultFS', 'hdfs://localhost:9000') \
    .getOrCreate()

HDFS_BASE = 'hdfs://localhost:9000/cs4074/lab5'

# Step 1: Read
df_dw = spark_dw.read.parquet(f'{HDFS_BASE}/data/yellow_tripdata_2024-01.parquet')

# Step 2: Filter (narrow)
df_filtered_dw = df_dw.filter(
    (col('fare_amount')   > 0) &
    (col('trip_distance') > 0)
)

# Step 3: Enrich (narrow)
df_enriched_dw = df_filtered_dw \
    .withColumn('tip_pct',
        round(col('tip_amount') / col('fare_amount') * 100, 2)) \
    .withColumn('trip_category',
        when(col('trip_distance') < 5,  'Short')
        .when(col('trip_distance') < 15, 'Medium')
        .otherwise('Long')) \
    .withColumn('pickup_hour', hour(col('tpep_pickup_datetime')))

# Step 4: Broadcast join (no shuffle for zones)
zone_schema_dw = StructType([
    StructField('LocationID',   IntegerType(), True),
    StructField('Borough',      StringType(),  True),
    StructField('Zone',         StringType(),  True),
    StructField('service_zone', StringType(),  True),
])
df_zones_dw = spark_dw.read \
    .option('header', 'true') \
    .schema(zone_schema_dw) \
    .csv(f'{HDFS_BASE}/data/taxi_zone_lookup.csv')

pickup_zones_dw = df_zones_dw \
    .withColumnRenamed('LocationID', 'PULocationID') \
    .withColumnRenamed('Borough',    'pickup_borough') \
    .select('PULocationID', 'pickup_borough')

df_joined_dw = df_enriched_dw.join(broadcast(pickup_zones_dw), 'PULocationID', 'left')
df_clean_dw  = df_joined_dw.fillna({'pickup_borough': 'Unknown', 'tip_pct': 0.0})

# Step 5: Aggregate (wide -- shuffle!)
summary_dw = df_clean_dw.groupBy('pickup_borough', 'trip_category').agg(
    count('*').alias('num_trips'),
    round(sum('total_amount'), 2).alias('total_revenue'),
    round(avg('fare_amount'),  2).alias('avg_fare'),
    round(avg('tip_pct'),      2).alias('avg_tip_pct'),
).orderBy(desc('total_revenue'))

# Step 6: Cache and display
summary_dw.cache()
summary_dw.show(20)

# Step 7: Write (triggers full DAG)
summary_dw.coalesce(1) \
    .write.mode('overwrite') \
    .option('header', 'true') \
    .csv(f'{HDFS_BASE}/output/deep_work_summary')

# Step 8: Plan inspection
summary_dw.explain(True)

# Cleanup
summary_dw.unpersist()
spark_dw.stop()

# Stop the main SparkSession
spark.stop()
print("All done. Lab 5 pipeline complete.")
