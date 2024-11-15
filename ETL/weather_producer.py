from pyspark.sql import SparkSession
from kafka import KafkaProducer
import json
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType


def process_weather_data(partition):
    # Create KafkaProducer inside the partition to avoid serialization issues
    producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    for row in partition:
        weather_data = {
            'datetime': row['datetime'].isoformat() if row['datetime'] else None,  # Convert datetime to string
            'tempmax': row['tempmax'],
            'tempmin': row['tempmin'],
            'humidity': row['humidity'],
            'precip': row['precip'],
            'snow': row['snow'],
            'windspeed': row['windspeed'],
            'windgust': row['windgust'],
            'conditions': row['conditions']
        }

        # Introduce sleep for rate limiting (simulate real-time data streaming)
        #time.sleep(1)

        # Send the weather data to Kafka
        producer.send('weather_events', weather_data)

    producer.flush()
    producer.close()

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("WeatherProducer") \
        .getOrCreate()

    # Define schema for weather data
    weather_schema = StructType([
        StructField("datetime", TimestampType(), True),
        StructField("tempmax", FloatType(), True),
        StructField("tempmin", FloatType(), True),
        StructField("humidity", FloatType(), True),
        StructField("precip", FloatType(), True),
        StructField("snow", FloatType(), True),
        StructField("windspeed", FloatType(), True),
        StructField("windgust", FloatType(), True),
        StructField("conditions", StringType(), True)
    ])

    df = spark.read.format("csv") \
        .schema(weather_schema) \
        .option("header", True) \
        .option("ignoreIndex", True) \
        .load("simulation data/weatherdata.csv")
    
    df.rdd.foreachPartition(process_weather_data)

    spark.stop()