from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, unix_timestamp
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from kafka import KafkaProducer
import json

# Initialize Spark session with Kafka support
spark = SparkSession.builder \
    .appName("BikeConsumer1") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Define schema for incoming data
schema = StructType([
    StructField("ride_id", StringType()),
    StructField("ride_type", StringType()),
    StructField("started_at", TimestampType()),
    StructField("ended_at", TimestampType()),
    StructField("start_station_name", StringType()),
    StructField("start_station_id", StringType()),
    StructField("end_station_name", StringType()),
    StructField("end_station_id", StringType()),
    StructField("start_lat", StringType()),
    StructField("start_lng", StringType()),
    StructField("end_lat", StringType()),
    StructField("end_lng", StringType()),
    StructField("member_casual", StringType())
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "bikeride") \
    .load()

# Parse the JSON data
parsed_df = df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# Calculate the duration in seconds
processed_df = parsed_df.withColumn("duration", 
                                    (unix_timestamp("ended_at") - unix_timestamp("started_at")).cast("double"))

# Filter for non-null duration
non_null_df = processed_df.filter(col("start_station_id").isNotNull())

# Convert data to JSON format to send to Kafka
result_df = non_null_df.selectExpr("to_json(struct(*)) AS value")

# Define a function to send each row to Kafka
def send_to_kafka(partition_data, partition_id=1):
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: v.encode('utf-8')
    )

    for row in partition_data.collect():
        ride_data = {'ride_id': row['ride_id'],
                    'started_at': str(row['started_at']),
                    'ride_type': str(row['ride_type']),
                    'ended_at': str(row['ended_at']),
                    'start_station_name': row['start_station_name'],
                    'start_station_id': row['start_station_id'],
                    'end_station_name': row['end_station_name'],
                    'end_station_id': row['end_station_id'],
                    'start_lat': row['start_lat'],
                    'start_lng': row['start_lng'],
                    'end_lat': row['end_lat'],
                    'end_lng': row['end_lng'],
                    'member_casual': row['member_casual']}

        producer.send('bikeride', value=ride_data, partition=partition_id)

    producer.flush()
    producer.close()

# Write each micro-batch back to Kafka with the specified partition
result_df.writeStream \
    .foreachBatch(lambda batch_df, _: send_to_kafka(batch_df, partition_id=1)) \
    .start() \
    .awaitTermination()