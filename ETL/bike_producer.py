from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from kafka import KafkaProducer
import random
import time
import json
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType

def send_to_kafka(rows):
    # Initialize KafkaProducer inside the function
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    for row in rows:
        ride_data = {
            'ride_id': row['ride_id'],
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
            'member_casual': row['member_casual'],
        }

        # Random sleep time between 0.2 and 1.5 seconds
        #random_sleep = random.uniform(1, 1.5)
        #time.sleep(random_sleep)
        
        # Send message to Kafka partition 0
        producer.send('bikeride', value=ride_data, partition=0)

    # Flush and close the producer after processing the partition
    producer.flush()
    producer.close()

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("KafkaProducerBike") \
        .getOrCreate()
    
    bike_schema = StructType([
        StructField("ride_id", StringType(), True),
        StructField("ride_type", StringType(), True),
        StructField("started_at", TimestampType(), True),
        StructField("ended_at", TimestampType(), True),
        StructField("start_station_name", StringType(), True),
        StructField("start_station_id", StringType(), True),
        StructField("end_station_name", StringType(), True),
        StructField("end_station_id", StringType(), True),
        StructField("start_lat", FloatType(), True),
        StructField("start_lng", FloatType(), True),
        StructField("end_lat", FloatType(), True),
        StructField("end_lng", FloatType(), True),
        StructField("member_casual", StringType(), True)
    ])

    # Read the CSV file into a DataFrame
    df = spark.read.format("csv") \
        .schema(bike_schema) \
        .option("header", True) \
        .option("ignoreIndex", True) \
        .load("simulation data/bikedata.csv")
    
    df = df.orderBy(col("started_at"), ascending=True)

    # Use foreachPartition to send data to Kafka
    df.rdd.foreachPartition(send_to_kafka)

    # Stop the Spark session
    spark.stop()
