import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(
    page_title= "Bikeshare Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'A personal project by abhaypai@vt.edu',
    }
)

px.set_mapbox_access_token(st.secrets['mapbox_token'])

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2.5rem;
                    padding-right: 2.5rem;
                }
        </style>
        """, unsafe_allow_html=True)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


st.cache_resource()
def get_weather_data():
    #conn = duckdb.connect('bikeshare_db.duckdb')
    #weather = conn.execute("SELECT * from weather_data LIMIT 57").df()
    weather = pd.read_csv("simulation_data/weatherdata.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather[["tempmax", "tempmin", "humidity", "precip", "snow", "windspeed", "windgust", "conditions"]] = weather[["tempmax", "tempmin", "humidity", "precip", "snow", "windspeed", "windgust", "conditions"]].round(2)
    #conn.close()

    return weather

weather = get_weather_data()

@st.cache_resource()
def fetch_rides():
    rides = pd.read_csv("simulation_data/sampledata.csv")
    rides[["started_at", "ended_at"]] =  pd.to_datetime(rides[["started_at", "ended_at"]])

    return rides


@st.cache_resource()
def model_fit():
    pastdata = pd.read_csv("simulation_data/pastdata.csv")
    X = pastdata[["tempmax", "tempmin", "humidity", "precip", "snow", "windspeed", "windgust"]]
    y = pastdata["count"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    r2 = round(r2*100, 1)
    return model, r2

def predict(row):
    model, r2 = model_fit()
    predicted = model.predict(row)

    return round(predicted[0],0).astype(int), r2

def calculate_earnings(df):
    # Define a nested function that applies the conditions to each row
    def earning_logic(row):
        if row['member_casual'] == 'casual':
            if row['rideable_type'] == 'classic_bike':
                return 1 + round(row['duration'], 0) * 0.05
            elif row['rideable_type'] == 'electric_bike':
                return 1 + round(row['duration'], 0) * 0.15
        elif row['member_casual'] == 'member':
            if row['rideable_type'] == 'classic_bike':
                # Only applies if duration is over 45 minutes
                extra_minutes = max(row['duration'] - 45, 0)
                return extra_minutes * 0.05
            elif row['rideable_type'] == 'electric_bike':
                return row['duration'] * 0.1
        return 0 

    df['revenue'] = df.apply(earning_logic, axis=1)
    return df

def get_ride_data(date):
    #conn = duckdb.connect('bikeshare_db.duckdb')
    #rides = conn.execute(f"SELECT * FROM ride_data WHERE CAST(started_at AS DATE) = '{date}' ORDER BY started_at ASC").df()
    rides = fetch_rides()
    rides = rides[rides['started_at'].date() == date]
    rides = calculate_earnings(rides)
    return rides

if 'today' not in st.session_state:
    st.session_state.today = pd.to_datetime('2024-01-01').date()

if 'is_playing' not in st.session_state:
    st.session_state.is_playing = True


maincol = st.columns([2,1,1], vertical_alignment="bottom")
maincol[0].title("Real-time Bikeshare dashboard")

# Control buttons
with maincol[1]:
    subcol = st.columns(3, gap='small')
    play_button = subcol[0].button("Play")
    pause_button = subcol[1].button("Pause")
    restart_button = subcol[2].button("Restart")

# Update session state based on button presses
if play_button:
    st.session_state.is_playing = True
if pause_button:
    st.session_state.is_playing = False
if restart_button:
    st.session_state.today = pd.to_datetime('2024-01-01').date()
    st.session_state.is_playing = True

# Create a placeholder for dynamic content
placeholder = st.empty()
map_placeholder = st.empty()
charts_placeholder = st.empty()

if st.session_state.is_playing:
    for idx, row in weather.iterrows():
        current_date = row['datetime'].date()

        if current_date < st.session_state.today:
            continue  # Skip past days to reach the saved `today` date

        st.session_state.today = current_date
        currentweather = row[["tempmax", "tempmin", "humidity", "precip", "snow", "windspeed", "windgust"]].to_frame().T
        predicted_today, accuracy = predict(currentweather)
        rides = get_ride_data(current_date)
        rides[['start_lat', 'start_lng', 'end_lat', 'end_lng']] = rides[['start_lat', 'start_lng', 'end_lat', 'end_lng']].astype(float)

        #DAY ELEMENTS
        with placeholder.container():
            titlecol = st.columns(4)
            titlecol[0].markdown(f"## Date: {current_date}")
            intradayrefresh = titlecol[2].slider("Intraday refresh (in seconds)", min_value=1.0, max_value=5.0, step=0.5, value=2.0, key = "slider1 - "+str(idx))
            dailyrefresh = titlecol[3].slider("Daily refresh (in seconds)", min_value=1.0, max_value=5.0, step=0.5, value=2.0, key = "slider2 - "+str(idx))
            with st.container(border=True):
                col1 = st.columns([1,1.6,1.4,0.8,0.8,0.8])
                col1[0].metric("Predicted Rides", value=str(predicted_today)+" rides", delta = str(accuracy)+" % model fit", delta_color="off")
                col1[1].metric("Conditions", value=row['conditions'])
                col1[2].metric("Temp. (High to Low)", value = str(round(row['tempmax'],2))+"°C to "+str(round(row['tempmin'],2))+"°C")
                col1[3].metric("Precipitation", value=str(round(row['precip'],2))+"mm")
                col1[4].metric("Wind Gusts", value=str(round(row['windgust'],2))+"m/s")
                col1[5].metric("Humidity", value=str(round(row['humidity'],2))+"%")

            first = rides.iloc[0].to_frame().T
            fig = px.scatter_mapbox(first, lat = 'start_lat', lon = 'start_lng', height=340)

            #NON-DISAPPEARING TIME ELEMENTS
            for i in range(0, len(rides), int(len(rides)/8)):
                ride = rides.iloc[i].to_frame().T
                cumul_rides = pd.concat([ride, rides.iloc[:i]], ignore_index=True).drop_duplicates()
                cumul_rides['started_at'] = pd.to_datetime(cumul_rides['started_at'])
                fig.update_layout(
                            mapbox_style='carto-positron',  
                            mapbox_center={"lat": 38.888, "lon": -77.021},
                            mapbox_zoom=9, 
                            mapbox_bearing=0,
                            mapbox_pitch=-10,
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            showlegend=False  # Show legend if needed
                        )

                fig.add_trace(go.Scattermapbox(
                        lat=list(cumul_rides['start_lat']),
                        lon=list(cumul_rides['start_lng']),
                        mode='markers',
                        marker=dict(size=5,color="skyblue"),
                        cluster=dict(enabled=True),
                        textfont=dict(color='white')
                    ))

                with map_placeholder.container():
                    currenttime = cumul_rides.iloc[-1]['started_at'].time()
                    st.markdown("#### Time: "+str(currenttime))
                    col2 = st.columns([1.3,0.85,1.2])
                    col2[0].plotly_chart(fig, key="map - "+str(current_date)+" - "+str(i))

                    max_stn_start = cumul_rides[['start_station_name', "ride_id"]].groupby('start_station_name').count().reset_index().sort_values(by='ride_id', ascending=False).iloc[0]
                    col2[0].write("Most frequent start station: "+max_stn_start["start_station_name"]+" ("+str(max_stn_start['ride_id'])+")")

                    max_stn_end = cumul_rides[['end_station_name', "ride_id"]].groupby('end_station_name').count().reset_index().sort_values(by='ride_id', ascending=False).iloc[0]
                    col2[0].write("Most frequent end station: "+max_stn_end["end_station_name"]+" ("+str(max_stn_end['ride_id'])+")")

                with charts_placeholder.container():

                    #Second Column
                    type_count = cumul_rides["rideable_type"].value_counts().reset_index()
                    type_count['axes'] = 1
                    ridesfig = px.bar(type_count, x='count', y='axes', barmode = 'stack', orientation='h', color='rideable_type', height=140,  title="Ride distribution", text_auto=True, color_discrete_sequence=['teal', 'springgreen'])
                    ridesfig.update_layout(margin=dict(l=20, r=20, t=25, b=0))  
                    ridesfig.update_xaxes(title_text="")
                    ridesfig.update_yaxes(title_text='', showticklabels=False)             
                    col2[1].plotly_chart(ridesfig, key="ridetype "+str(current_date)+": "+str(i))

                    cumul_rides['time'] = cumul_rides['started_at'].dt.floor('1h')

                    # Calculate the average duration for each 3-hour interval
                    average_duration = cumul_rides[['time', 'ride_id', 'rideable_type']].groupby(['time','rideable_type']).count().reset_index()
                    ridetimefig = px.bar(average_duration, x="time", y='ride_id', title="Hourwise Ride Frequency", barmode='stack', color='rideable_type', height = 285, color_discrete_sequence=['teal', 'springgreen'])
                    ridetimefig.update_layout(margin=dict(l=20, r=20, t=25, b=30), showlegend=False, yaxis_showgrid=False)
                    ridetimefig.add_hline(
                                    y=average_duration['ride_id'].mean(),
                                    line_dash="dot",
                                    line_color="red",
                                    annotation_text=f"Average: {average_duration['ride_id'].mean():.0f} rides",
                                    annotation_position="top left",
                                    annotation_font_color="black"
                                )
                    col2[1].plotly_chart(ridetimefig, key="ridetime "+str(current_date)+": "+str(i))

                    #Third Column
                    revenue = round(cumul_rides['revenue'].sum(),2)
                    agg_revenue = cumul_rides[['rideable_type', 'revenue']].groupby('rideable_type').sum().reset_index()
                    agg_revenue['axes'] = 1
                    with col2[2]:
                        row1 = st.columns([1,0.55], vertical_alignment='top')

                        revfig = px.bar(agg_revenue, x='revenue', y='axes', barmode='stack', orientation='h', color = 'rideable_type', title="Revenue distribution", height=140, text_auto=True, color_discrete_sequence=['teal', 'springgreen'])
                        revfig.update_yaxes(title_text='', showticklabels=False)
                        revfig.update_xaxes(title_text="")  
                        revfig.update_layout(margin=dict(l=10, r=20, t=28, b=20), showlegend=False)
                        row1[0].plotly_chart(revfig, key="revfig "+str(current_date)+": "+str(i))

                        row1[1].metric("Revenue collected today", value = "$"+str(revenue))

                        row2 = st.columns([1,0.96,1])

                        rev_per_ride = pd.merge(agg_revenue.drop(columns="axes"), type_count, on="rideable_type", how="inner")
                        rev_per_ride['rpr'] = rev_per_ride['revenue']/rev_per_ride['count']

                        if len(cumul_rides) > 1:
                            classic_rpr = rev_per_ride.loc[rev_per_ride['rideable_type']=="classic_bike", 'rpr'].values[0]
                            electric_rpr = rev_per_ride.loc[rev_per_ride['rideable_type']=="electric_bike", 'rpr'].values[0]
                            
                            row2[0].metric("Rides completed today", value = i+1, delta = str(i+1-predicted_today)+" (Predicted)")
                            row2[1].metric("Revenue/ride (Classic)", value="$"+str(round(classic_rpr,2)))
                            row2[2].metric("Revenue/ride (Electric)", value="$"+str(round(electric_rpr,2)))


                time.sleep(intradayrefresh)
            

        time.sleep(dailyrefresh)
