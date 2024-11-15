import pandas as pd
from kafka import KafkaConsumer
import duckdb
import json

# Kafka consumer configuration
consumer = KafkaConsumer('weather_events', bootstrap_servers=['localhost:9092'],
                          value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# Connect to DuckDB
con = duckdb.connect('bikeshare_db.duckdb')

batch = []
batch_size = 20  # Define batch size

try:
    # Consume messages continuously
    for message in consumer:
        weather_data = message.value
        batch.append(weather_data)  # Add message to batch

        # If batch is full, write to DuckDB and clear the batch
        if len(batch) >= batch_size:
            df = pd.DataFrame(batch)
            # Avoid passing DuckDB connection inside a distributed function
            df.to_sql("weather_data", con, if_exists="append", index=False)
            batch = []  # Clear the batch

except KeyboardInterrupt:
    print("Consumer interrupted manually.")
finally:
    # Write any remaining messages in batch
    if batch:
        df = pd.DataFrame(batch)
        df.to_sql("weather_data", con, if_exists="append", index=False)

    # Close DuckDB connection and consumer
    con.commit()
    con.close()
    consumer.close()

print("Consumer stopped.")