from kafka import KafkaConsumer
import json
import duckdb

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'bikeride',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',  # Read only new messages
    enable_auto_commit=True,
    group_id='bike_event_consumers',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Connect to DuckDB and create the table if it doesn't exist
con = duckdb.connect('bikeshare_db.duckdb')

# Batch insert buffer
batch_size = 30  # Number of messages to collect before a batch insert
batch_data = []

# Consume messages from Kafka
for message in consumer:
    data = message.value

    record = (
        data.get('ride_id'),
        data.get('started_at'),
        data.get('ended_at'),
        data.get('start_station_name'),
        data.get('start_station_id'),
        data.get('end_station_name'),
        data.get('end_station_id'),
        data.get('start_lat'),
        data.get('start_lng'),
        data.get('end_lat'),
        data.get('end_lng'),
        data.get('member_casual'),
        data.get('duration')
    )

    # Add the record to the batch
    batch_data.append(record)

# Insert batch into DuckDB if batch size is reached
if len(batch_data) >= batch_size:
    con.executemany("INSERT INTO ride_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_data)
    batch_data.clear()  # Clear the batch after inserting

# Final insert for any remaining records in the batch
if batch_data:
    con.executemany("INSERT INTO ride_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_data)
    batch_data.clear()

# Close the DuckDB connection       
con.close()