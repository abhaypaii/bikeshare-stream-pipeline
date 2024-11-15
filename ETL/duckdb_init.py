import duckdb

con = duckdb.connect('bikeshare_db.duckdb')

con.execute("""
    CREATE TABLE IF NOT EXISTS weather_data (
        datetime TIMESTAMP PRIMARY KEY,
        tempmax FLOAT,
        tempmin FLOAT,
        humidity FLOAT,
        precip FLOAT,
        snow FLOAT,
        windspeed FLOAT,
        windgust FLOAT,
        conditions TEXT
    )
""")

con.execute("""
    CREATE TABLE IF NOT EXISTS ride_data (
        ride_id TEXT PRIMARY KEY,
        rideable_type TEXT,
        started_at TIMESTAMP,
        ended_at TIMESTAMP,
        start_station_name TEXT,
        start_station_id TEXT,
        end_station_name TEXT,
        end_station_id TEXT,
        start_lat TEXT,
        start_lng TEXT,
        end_lat TEXT,
        end_lng TEXT,
        member_casual TEXT,
        duration DOUBLE
    )
""")

con.close()