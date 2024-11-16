
# Event Streaming Pipeline Simulation using Kafka, Spark and DuckDB

This pipeline simulates real-time event streaming of customers riding Capital Bikeshare bikes in Washington DC.

I have used a Kaggle dataset of Capital Bikeshare rides in D.C. from 2020 to 2024, with over 9 million rows, to simulate event streaming using Apache Kafka. I also use Apache Spark to execute distributed processing of big data and I load the data onto my local DuckDB. Finally, I leverage Streamlit to create simulate real-time daily bike ride event streaming with insightful KPIs alongside Random Forest Regression on the daily weather data to forecast ride volume on a day-to-day basis. This allows to provide more granular information to the stakehodlers and decision makers

## Data Source
(https://www.kaggle.com/datasets/taweilo/capital-bikeshare-dataset-202005202408?select=daily_rent_detail.csv)

## Softwares used

Python version: 3.11.1

    1. Apache Kafka for Event streaming
    2. Apache Spark Structured Streams for real-time data processing.
    3. DuckDB for data warehousing.
    4. Scikit-learn for Forecasting based on weather data.
    5. Streamlit for interactive data viz. 
## Pipeline Overview

<img src="images/system design.png" alt="Pipeline" width="1000"/>

## File Directory

    .
    ├── .streamlit/
    │   ├── config.toml
    │   └── secrets.toml
    ├── ETL/
    │   ├── bike_producer.py
    │   ├── bike_transform.py
    │   ├── bike_consumer.py
    │   ├── weather_producer.py
    │   ├── weather_consumer.py
    │   └── duckdb_init.py
    ├── app.py
    ├── bikeshare_db.duckdb
    ├── preprocessing.py
    ├── .gitignore
    └── requirements.txt

## System Setup

1. Install Kafka and Zookeeper onto your system:

        brew install zookeeper
        brew install kafka

After you go into your project directory, create a virtual environment and install these packages

2. Install Kafka's python client

        pip install kafka-python
    

3. Install PySpark:

        pip install pyspark

4. Install duckdb:

        pip install duckdb

5. Install streamlit:

        pip install streamlit
## Initialising Kafka topics

Start Zookeeper on a terminal window:

    zookeeper-server-start /opt/homebrew/etc/kafka/zookeeper.properties

Start Kafka in a different shell:

    /opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties

Create a Kafka topic for ride events in a different window:

    kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 2 --topic bikerides

Create a Kafka topic for daily weather in a different window:

    kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic weather
## ETL Overview

* bike_producer.py: Kafka Producer to send bike ride events to the bikeride topic.

* bike_transform.py: Spark Structured Streaming to ingest real-time data from the first partition of the bikeride topic, clean the data by reducing null values, adds a column for total ride duration and sends the transformed data back to the second partition of the same topic

* bike_consumer.py: Ingests data from the second partition of bikerid topic and stores it in ride_data table of bikeshare_db.duckdb.

* weather_producer.py: Sends daily weather data to the weather topic.

* weather_consumer.py: Consumes daily weather data to store it in weather_data table of bikeshare_db.duckdb.

* duckdb_init.py: Python file to create tables in the duckdb file.
