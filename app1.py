import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import io
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import time
import math
from scipy.stats import pearsonr

# Set page configuration
st.set_page_config(
    page_title="Earthquake Monitoring Dashboard",
    page_icon="🌎",
    layout="wide"
)

# Title and description
st.title("🌎 Earthquake Monitoring Dashboard")
st.write("Explore earthquake data and predictions using machine learning models.")
st.markdown("""
This dashboard helps you explore earthquake data with easy-to-understand visualizations and prediction tools(based on data).
Data comes from the USGS Earthquake Catalog, which is the official source for earthquake information in the United States .
""")
st.markdown("Developed by: **Dr. Anil Kumar Singh**")
st.markdown("Email: **singhanil854@gmail.com**")

# Add a help button in the top corner
with st.expander("❓ How to Use This Dashboard"):
    st.markdown("""
    ### Getting Started
    This dashboard is designed to be easy to use for everyone, regardless of technical background:
    
    1. **Select your data**: Use the sidebar on the left to choose the time period, earthquake size range, and region you want to explore.
    2. **Explore the tabs**: The dashboard is organized into different sections that you can access by clicking on the tabs.
    3. **Interact with visualizations**: Hover over charts to see details, zoom in/out on maps, and adjust filters to customize your view.
    
    ### What You Can Do
    - See where earthquakes are happening on interactive maps
    - Explore patterns in earthquake activity over time
    - Understand relationships between earthquake characteristics
    - Learn about possible future earthquake activity through educational prediction models
    
    ### Important Note
    While this dashboard uses scientific data and methods, precise earthquake prediction remains challenging. The predictions shown are for educational purposes and should not be used for safety planning.
    """)

# Sidebar for user inputs
st.sidebar.header("Data Selection")

# Help tooltip for data parameters
with st.sidebar.expander("ℹ️ Help: Selecting Data"):
    st.markdown("""
    **Date Range**: Choose the time period you want to analyze. A wider range gives you more data but might be slower to load.
    
    **Magnitude Range**: Earthquakes are measured on the Richter scale:
    - 0-2: Usually not felt but recorded
    - 3-4: Often felt, minimal damage
    - 5-6: Slight to moderate damage
    - 7+: Major to catastrophic damage
    
    **Region**: Select a geographical area to focus on. "Worldwide" shows all earthquakes.
    """)

# Date range selection
today = datetime.now()
default_start_date = today - timedelta(days=30)

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", today)

# Magnitude range
min_magnitude = st.sidebar.slider("Minimum Magnitude", 0.0, 9.0, 4.0)
max_magnitude = st.sidebar.slider("Maximum Magnitude", 0.0, 9.0, 9.0)

# Region selection
region_options = {
    "Worldwide": {"minlatitude": -90, "maxlatitude": 90, "minlongitude": -180, "maxlongitude": 180},
    "North America": {"minlatitude": 15, "maxlatitude": 72, "minlongitude": -170, "maxlongitude": -50},
    "South America": {"minlatitude": -60, "maxlatitude": 15, "minlongitude": -90, "maxlongitude": -30},
    "Europe": {"minlatitude": 35, "maxlatitude": 75, "minlongitude": -10, "maxlongitude": 40},
    "Asia": {"minlatitude": 0, "maxlatitude": 60, "minlongitude": 60, "maxlongitude": 150},
    "Africa": {"minlatitude": -40, "maxlatitude": 40, "minlongitude": -20, "maxlongitude": 55},
    "Australia & Oceania": {"minlatitude": -50, "maxlatitude": 0, "minlongitude": 110, "maxlongitude": 180},
    "Japan & East Asia": {"minlatitude": 20, "maxlatitude": 50, "minlongitude": 120, "maxlongitude": 150},
    "Mediterranean": {"minlatitude": 30, "maxlatitude": 45, "minlongitude": -5, "maxlongitude": 40},
    "Pacific Ring of Fire": {"minlatitude": -40, "maxlatitude": 60, "minlongitude": 120, "maxlongitude": -120},
    "California": {"minlatitude": 32, "maxlatitude": 42, "minlongitude": -125, "maxlongitude": -114},
}

selected_region = st.sidebar.selectbox("Select Region", list(region_options.keys()))
region_coords = region_options[selected_region]

# Function to fetch earthquake data from USGS
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_earthquake_data(start_date, end_date, min_magnitude, max_magnitude, region_coords):
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    params = {
        "format": "csv",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),  # Add one day to include end date
        "minmagnitude": min_magnitude,
        "maxmagnitude": max_magnitude,
        "minlatitude": region_coords["minlatitude"],
        "maxlatitude": region_coords["maxlatitude"],
        "minlongitude": region_coords["minlongitude"],
        "maxlongitude": region_coords["maxlongitude"],
        "orderby": "time"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        
        if response.text.strip():  # Check if response is not empty
            data = pd.read_csv(io.StringIO(response.text))
            return data
        else:
            return pd.DataFrame()
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load data with progress indicator
with st.spinner('Fetching earthquake data...'):
    df = fetch_earthquake_data(start_date, end_date, min_magnitude, max_magnitude, region_coords)

# Check if data was successfully loaded
if df.empty:
    st.warning("No earthquake data found for the selected criteria. Try adjusting your parameters.")
    st.info("Suggestions: Try a wider date range, lower the minimum magnitude, or select a different region.")
    st.stop()

# Data preprocessing
try:
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Extract date components
    df['date'] = df['time'].dt.date
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['weekday'] = df['time'].dt.day_name()
    
    # Round coordinates for clustering
    df['lat_rounded'] = np.round(df['latitude'], 1)
    df['lon_rounded'] = np.round(df['longitude'], 1)
    
    # Create magnitude categories
    def categorize_magnitude(mag):
        if mag < 2.0:
            return "Minor (< 2.0)"
        elif mag < 4.0:
            return "Light (2.0-4.0)"
        elif mag < 5.0:
            return "Moderate (4.0-5.0)"
        elif mag < 6.0:
            return "Strong (5.0-6.0)"
        elif mag < 7.0:
            return "Major (6.0-7.0)"
        else:
            return "Great (7.0+)"
    
    df['magnitude_category'] = df['mag'].apply(categorize_magnitude)
    
    # Estimate energy release in joules (using formula: E = 10^(1.5*M + 4.8))
    df['energy_joules'] = 10 ** (1.5 * df['mag'] + 4.8)
    
    # Convert to equivalent tons of TNT (1 ton TNT = 4.184 * 10^9 joules)
    df['energy_tnt'] = df['energy_joules'] / (4.184 * 10**9)
    
    # Add tsunami potential indicator (simple heuristic - coastal quakes > 6.5)
    coastal_depth = 100  # km from coastline approximation
    df['tsunami_potential'] = ((df['mag'] >= 6.5) & (df['depth'] <= coastal_depth)).astype(int)

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.write("Raw data preview:")
    st.write(df.head())
    st.stop()

# Main dashboard content
st.header("Earthquake Data Overview")

# Display basic statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Earthquakes", df.shape[0], help="Total number of earthquakes matching your criteria")
with col2:
    st.metric("Average Magnitude", round(df['mag'].mean(), 2), help="Average earthquake size on the Richter scale")
with col3:
    st.metric("Strongest Earthquake", round(df['mag'].max(), 2), help="Magnitude of the largest earthquake in this dataset")
with col4:
    energy_sum = df['energy_tnt'].sum() / 1000  # Convert to kilotons
    st.metric("Total Energy Released", f"{energy_sum:.2f} kilotons TNT", 
              help="Total seismic energy released (equivalent to kilotons of TNT)")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Map View", 
    "📊 Time Analysis", 
    "🔮 Predictions", 
    "🔍 Risk Analysis",
    "📱 Mobile Alert System",
    "📋 Data Explorer"
])

# Tab 1: Map View
with tab1:
    st.subheader("Geographic Distribution of Earthquakes")
    
    with st.expander("ℹ️ Help: Map View"):
        st.markdown("""
        **What am I looking at?**
        - Each dot on the map represents an earthquake
        - Larger, darker dots indicate stronger earthquakes
        - You can click on any earthquake to see its details
        - You can zoom in/out and drag the map to explore
        
        **The Heatmap** shows areas with many earthquakes, regardless of their strength.
        
        **The Cluster Analysis** groups nearby earthquakes to identify active zones.
        """)
    
    # Map plot with Plotly
    fig_map = px.scatter_mapbox(
        df, 
        lat="latitude", 
        lon="longitude", 
        color="mag",
        size="mag",
        color_continuous_scale=px.colors.sequential.Inferno,
        size_max=15,
        zoom=1,
        hover_name="place",
        hover_data={
            "mag": True, 
            "depth": True, 
            "time": True, 
            "latitude": False, 
            "longitude": False,
            "energy_tnt": ":.2f"
        },
        labels={"energy_tnt": "Energy (tons TNT)"},
        mapbox_style="carto-positron",
        title="Earthquake Locations"
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Heatmap of earthquake frequency by location
    st.subheader("Earthquake Frequency Heat Map")
    
    location_counts = df.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
    
    fig_heatmap = px.density_mapbox(
        location_counts,
        lat='lat_rounded',
        lon='lon_rounded',
        z='count',
        radius=20,
        center=dict(lat=0, lon=0),
        zoom=1,
        mapbox_style="carto-positron",
        color_continuous_scale=px.colors.sequential.Inferno
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Cluster analysis to identify seismic zones
    st.subheader("Earthquake Cluster Analysis")
    
    with st.expander("ℹ️ What is Cluster Analysis?"):
        st.markdown("""
        Cluster analysis helps identify groups of earthquakes that might be related to the same geological feature or event.
        
        - Each color represents a group of earthquakes that are geographically close to each other
        - This can help identify active fault lines and seismic zones
        - The number of clusters is automatically determined based on your data
        """)
    
    # Determine optimal number of clusters based on data size
    n_clusters = min(max(3, int(df.shape[0] / 50)), 10)  # Between 3 and 10 clusters
    
    # Prepare data for clustering
    coords = df[['latitude', 'longitude']].copy()
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # Create cluster centers dataframe
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
    cluster_centers['cluster'] = range(n_clusters)
    
    # Count earthquakes in each cluster
    cluster_counts = df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    
    # Merge counts with centers
    cluster_centers = pd.merge(cluster_centers, cluster_counts, on='cluster')
    
    # Calculate average magnitude per cluster
    cluster_mag = df.groupby('cluster')['mag'].mean().reset_index()
    cluster_mag.columns = ['cluster', 'avg_magnitude']
    cluster_centers = pd.merge(cluster_centers, cluster_mag, on='cluster')
    
    # Create the cluster map
    fig_clusters = px.scatter_mapbox(
        df, 
        lat="latitude", 
        lon="longitude", 
        color="cluster",
        color_continuous_scale=px.colors.qualitative.Bold,
        hover_name="place",
        hover_data=["mag", "depth", "time"],
        mapbox_style="carto-positron",
        zoom=1,
        title="Earthquake Clusters"
    )
    
    # Add cluster centers
    fig_clusters.add_trace(
        go.Scattermapbox(
            lat=cluster_centers['latitude'],
            lon=cluster_centers['longitude'],
            mode='markers',
            marker=dict(size=20, color='white', opacity=0.8),
            text=["Cluster " + str(i) + "<br>" + 
                  str(row['count']) + " earthquakes<br>" +
                  "Avg mag: " + str(round(row['avg_magnitude'], 2))
                  for i, row in cluster_centers.iterrows()],
            hoverinfo='text',
            name='Cluster Centers'
        )
    )
    
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    # Display top seismic regions
    st.subheader("Most Active Regions")

    # Group data by place
    place_stats = df.groupby('place').agg(
        count=('mag', 'count'),
        avg_mag=('mag', 'mean'),
        max_mag=('mag', 'max'),
        last_event=('time', 'max')
    ).reset_index()
    
    # Sort by count and get top 10
    top_places = place_stats.sort_values('count', ascending=False).head(10)
    
    # Create the bar chart
    fig_places = px.bar(
        top_places,
        x='place',
        y='count',
        color='avg_mag',
        color_continuous_scale=px.colors.sequential.Inferno,
        hover_data=['max_mag', 'last_event'],
        labels={
            'place': 'Location', 
            'count': 'Number of Earthquakes',
            'avg_mag': 'Average Magnitude',
            'max_mag': 'Maximum Magnitude',
            'last_event': 'Most Recent Event'
        },
        title="Top 10 Most Active Earthquake Regions"
    )
    
    fig_places.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_places, use_container_width=True)

# Tab 2: Time Analysis
with tab2:
    st.subheader("Temporal Analysis")
    
    with st.expander("ℹ️ Help: Understanding Temporal Patterns"):
        st.markdown("""
        **What am I looking at?**
        - These charts show how earthquake activity changes over time
        - The bars show how many earthquakes occurred each day
        - The red line shows the average magnitude
        
        **Why This Matters:**
        - Clusters of earthquakes may indicate aftershocks or triggered events
        - Some patterns may repeat over time
        - Understanding these patterns helps scientists study earthquake behavior
        """)
    
    # Time series of earthquake occurrences
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_avg_mag = df.groupby('date')['mag'].mean().reset_index(name='avg_magnitude')
    daily_max_mag = df.groupby('date')['mag'].max().reset_index(name='max_magnitude')
    daily_energy = df.groupby('date')['energy_tnt'].sum().reset_index(name='energy_tnt')
    
    # Merge all daily stats
    daily_data = pd.merge(daily_counts, daily_avg_mag, on='date')
    daily_data = pd.merge(daily_data, daily_max_mag, on='date')
    daily_data = pd.merge(daily_data, daily_energy, on='date')
    
    # Add 3-day moving average
    daily_data['count_ma3'] = daily_data['count'].rolling(window=3, min_periods=1).mean()
    
    # Create the time series plot
    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_time.add_trace(
        go.Bar(
            x=daily_data['date'], 
            y=daily_data['count'], 
            name="Number of Earthquakes", 
            marker_color='darkblue',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    fig_time.add_trace(
        go.Scatter(
            x=daily_data['date'], 
            y=daily_data['count_ma3'], 
            name="3-Day Moving Average", 
            marker_color='blue',
            line=dict(width=2)
        ),
        secondary_y=False
    )
    
    fig_time.add_trace(
        go.Scatter(
            x=daily_data['date'], 
            y=daily_data['avg_magnitude'], 
            name="Average Magnitude", 
            marker_color='red',
            line=dict(width=2)
        ),
        secondary_y=True
    )
    
    fig_time.update_layout(
        title="Daily Earthquake Frequency and Magnitude",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_time.update_yaxes(title_text="Number of Earthquakes", secondary_y=False)
    fig_time.update_yaxes(title_text="Average Magnitude", secondary_y=True)
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Energy release over time
    st.subheader("Daily Energy Release")
    
    fig_energy = px.bar(
        daily_data,
        x='date',
        y='energy_tnt',
        labels={
            'date': 'Date', 
            'energy_tnt': 'Energy Released (tons of TNT)'
        },
        title="Daily Seismic Energy Release"
    )
    
    st.plotly_chart(fig_energy, use_container_width=True)
    
    # Earthquake distribution by hour of day
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Earthquakes by Hour of Day")
        hourly_counts = df.groupby('hour').size().reset_index(name='count')
        hourly_avg = df.groupby('hour')['mag'].mean().reset_index(name='avg_magnitude')
        hourly_data = pd.merge(hourly_counts, hourly_avg, on='hour')
        
        fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_hourly.add_trace(
            go.Bar(
                x=hourly_data['hour'], 
                y=hourly_data['count'], 
                name="Number of Earthquakes",
                marker_color='darkblue'
            ),
            secondary_y=False
        )
        
        fig_hourly.add_trace(
            go.Scatter(
                x=hourly_data['hour'], 
                y=hourly_data['avg_magnitude'], 
                name="Average Magnitude",
                marker_color='red'
            ),
            secondary_y=True
        )
        
        fig_hourly.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=4),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_hourly.update_yaxes(title_text="Number of Earthquakes", secondary_y=False)
        fig_hourly.update_yaxes(title_text="Average Magnitude", secondary_y=True)
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.subheader("Earthquakes by Day of Week")
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = df.groupby('weekday').size().reset_index(name='count')
        weekday_avg = df.groupby('weekday')['mag'].mean().reset_index(name='avg_magnitude')
        weekday_data = pd.merge(weekday_counts, weekday_avg, on='weekday')
        
        # Reorder days
        weekday_data['weekday'] = pd.Categorical(weekday_data['weekday'], categories=weekday_order, ordered=True)
        weekday_data = weekday_data.sort_values('weekday')
        
        fig_weekday = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_weekday.add_trace(
            go.Bar(
                x=weekday_data['weekday'], 
                y=weekday_data['count'], 
                name="Number of Earthquakes",
                marker_color='darkblue'
            ),
            secondary_y=False
        )
        
        fig_weekday.add_trace(
            go.Scatter(
                x=weekday_data['weekday'], 
                y=weekday_data['avg_magnitude'], 
                name="Average Magnitude",
                marker_color='red'
            ),
            secondary_y=True
        )
        
        fig_weekday.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_weekday.update_yaxes(title_text="Number of Earthquakes", secondary_y=False)
        fig_weekday.update_yaxes(title_text="Average Magnitude", secondary_y=True)
        
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Magnitude distribution
    st.subheader("Magnitude Distribution")
    
    with st.expander("ℹ️ Understanding Earthquake Magnitude"):
        st.markdown("""
        **What is magnitude?**
        - Magnitude measures the energy released by an earthquake
        - The scale is logarithmic - each whole number increase means about 32 times more energy
        - A magnitude 7.0 earthquake releases 32 times more energy than a 6.0 and 1,000 times more than a 5.0
        
        **What does this chart show?**
        - This histogram shows how many earthquakes occurred at each magnitude level
        - There are usually many more small earthquakes than large ones
        - The distribution pattern can help scientists understand the seismic behavior of a region
        """)
    
    fig_mag_dist = px.histogram(
        df,
        x="mag",
        nbins=50,
        labels={"mag": "Magnitude", "count": "Frequency"},
        title="Earthquake Magnitude Distribution"
    )
    
    # Add a vertical line for average magnitude
    fig_mag_dist.add_vline(
        x=df['mag'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {df['mag'].mean():.2f}",
        annotation_position="top right"
    )
    
    st.plotly_chart(fig_mag_dist, use_container_width=True)
    
    # Depth vs Magnitude scatter plot
    st.subheader("Depth vs. Magnitude Relationship")
    
    with st.expander("ℹ️ Understanding Earthquake Depth"):
        st.markdown("""
        **What is earthquake depth?**
        - Depth is how far below the Earth's surface an earthquake occurs
        - Shallow earthquakes (< 70 km) can cause more surface damage
        - Deep earthquakes (> 300 km) often cause less damage but can be felt over larger areas
        
        **What am I looking at?**
        - Each dot represents an earthquake
        - The x-axis shows depth in kilometers
        - The y-axis shows magnitude
        - Look for patterns - do stronger earthquakes tend to be deeper or shallower in this region?
        """)
    
    fig_depth_mag = px.scatter(
        df,
        x="depth",
        y="mag",
        color="mag",
        size="mag",
        color_continuous_scale=px.colors.sequential.Inferno,
        labels={"depth": "Depth (km)", "mag": "Magnitude"},
        title="Earthquake Depth vs. Magnitude",
        hover_data=["place", "time"]
    )
    
    # Add trend line
    fig_depth_mag.add_trace(
        go.Scatter(
            x=[df['depth'].min(), df['depth'].max()],
            y=np.poly1d(np.polyfit(df['depth'], df['mag'], 1))(
                [df['depth'].min(), df['depth'].max()]
            ),
            mode='lines',
            line=dict(color='white', dash='dash'),
            name='Trend Line'
        )
    )
    
    # Add depth classification regions
    if df['depth'].max() > 70:
        fig_depth_mag.add_vrect(
            x0=0, x1=70,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Shallow",
            annotation_position="top left"
        )
    
    if df['depth'].max() > 70 and df['depth'].min() < 300:
        fig_depth_mag.add_vrect(
            x0=70, x1=300,
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Intermediate",
            annotation_position="top left"
        )
    
    if df['depth'].max() > 300:
        fig_depth_mag.add_vrect(
            x0=300, x1=df['depth'].max() + 10,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Deep",
            annotation_position="top left"
        )
    
    st.plotly_chart(fig_depth_mag, use_container_width=True)
    
    # Calculate correlation
    depth_mag_corr, depth_mag_p = pearsonr(df['depth'], df['mag'])
    
    st.info(f"""
    **Correlation Analysis**: The correlation between depth and magnitude is {depth_mag_corr:.3f}
    ({depth_mag_p:.3f} p-value).
    
    {'This suggests a statistically significant relationship between depth and magnitude in this dataset.' 
     if depth_mag_p < 0.05 else 
     'No statistically significant correlation was found between depth and magnitude in this dataset.'}
    """)

# Tab 3: Prediction Models
with tab3:
    st.subheader("Earthquake Prediction Models")
    
    with st.expander("ℹ️ Understanding Prediction Limitations"):
        st.markdown("""
        **Important Note About Earthquake Prediction**
        
        Predicting exactly when and where an earthquake will occur is not currently possible with scientific certainty.
        
        What these models DO:
        - Identify patterns in past earthquake data
        - Show relationships between factors like location, depth, and magnitude
        - Help understand which areas may be more seismically active in general
        
        What these models DON'T do:
        - Predict specific future earthquakes with certainty
        - Replace official warnings or forecasts from geological agencies
        - Account for all possible geological factors
        
        This section is provided for educational purposes to illustrate how data science approaches earthquake analysis.
        """)
    
    st.markdown("""
    These models analyze patterns in historical earthquake data to identify relationships between location, depth, time factors, 
    and earthquake magnitude. While not definitive predictors of future events, they help us understand patterns in seismic activity.
    """)
    
    # Check if we have enough data for prediction
    if df.shape[0] > 20:  # Minimum threshold for meaningful prediction
        # Feature selection for prediction
        features = ['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour', 'dayofweek']
        
        # Prepare data for magnitude prediction
        X = df[features]
        y = df['mag']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model options
        model_option = st.selectbox(
            "Select Prediction Model", 
            ["Random Forest", "Gradient Boosting"],
            help="Different models use different techniques to find patterns in earthquake data"
        )
        
        # Model training
        with st.spinner('Training prediction model...'):
            if model_option == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model_name = "Gradient Boosting"
                
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        
        # Display model performance
        st.subheader(f"Model Performance: {model_name}")
        
        with st.expander("ℹ️ Understanding Model Metrics"):
            st.markdown("""
            **What do these numbers mean?**
            
            - **Mean Absolute Error (MAE)**: Average difference between predicted and actual magnitude values
            - **Root Mean Squared Error (RMSE)**: Similar to MAE but penalizes large errors more heavily
            - **R² Score**: How well the model explains variation in earthquake magnitudes (0-1 scale)
                - 0 = no explanatory power
                - 1.0 = perfect predictions
                - Higher is better
            
            For earthquake magnitude prediction, MAE and RMSE below 0.5 are considered reasonably good.
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.3f}", help="Average error in magnitude predictions")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:.3f}", help="Error with higher penalty for large mistakes")
        with col3:
            st.metric("R² Score", f"{r2:.3f}", help="Model accuracy (0-1 scale, higher is better)")
        
        # Feature importance
        st.subheader("What Factors Influence Earthquake Magnitude?")
        
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        with st.expander("ℹ️ Understanding Feature Importance"):
            st.markdown("""
            **What is this chart showing?**
            
            This chart shows which factors have the strongest relationship with earthquake magnitude according to our model.
            
            - **Higher bars** mean that factor has more influence on earthquake magnitude
            - For example, if depth has a high importance, it means deeper or shallower earthquakes tend to have different magnitudes
            - This can help us understand which geological factors might play a role in earthquake strength
            """)
        
        fig_importance = px.bar(
            feature_importance,
            x='Feature',
            y='Importance',
            title=f"Feature Importance for Earthquake Magnitude ({model_name})"
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Actual vs Predicted plot
        st.subheader("How Accurate are the Predictions?")
        
        with st.expander("ℹ️ How to Read This Chart"):
            st.markdown("""
            **What am I looking at?**
            
            - Each dot represents an earthquake
            - The x-axis shows the actual measured magnitude
            - The y-axis shows what our model predicted for that earthquake
            - Dots on the red dashed line = perfect predictions
            - Dots above the line = model overestimated magnitude
            - Dots below the line = model underestimated magnitude
            
            The closer the dots are to the red line, the more accurate our model.
            """)
        
        fig_pred = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual Magnitude", "y": "Predicted Magnitude"},
            title="Actual vs. Predicted Earthquake Magnitude"
        )
        
        fig_pred.add_trace(
            go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                      mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash'))
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Cross-validation for robustness
        st.subheader("Model Stability Check")
        
        with st.expander("ℹ️ What is Cross-Validation?"):
            st.markdown("""
            Cross-validation tests the model on different subsets of data to ensure it performs consistently.
            
            - We split the data into 5 parts
            - Train the model 5 times, each time leaving out one part
            - Test on the part we left out
            - This gives us 5 different accuracy scores
            
            If the scores are similar, our model is stable and not overly dependent on which specific earthquakes were in the training data.
            """)
        
        with st.spinner("Performing cross-validation..."):
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            cv_mae_scores = -cv_scores  # Convert from negative to positive MAE
        
        # Display cross-validation results
        cv_df = pd.DataFrame({
            'Fold': range(1, 6),
            'MAE': cv_mae_scores
        })
        
        fig_cv = px.bar(
            cv_df,
            x='Fold',
            y='MAE',
            title="Cross-Validation Results (Mean Absolute Error)"
        )
        
        fig_cv.add_hline(
            y=cv_mae_scores.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {cv_mae_scores.mean():.3f}",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
        
        # Interactive prediction
        st.header("Predict Earthquake Magnitude")
        st.markdown("""
        Adjust the parameters below to see what magnitude our model would predict for an earthquake with these characteristics.
        Remember, this is an educational tool to explore patterns, not an actual forecasting system.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            pred_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
            pred_long = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
            pred_depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0)
        
        with col2:
            pred_date = st.date_input("Date", datetime.now())
            pred_hour = st.slider("Hour of Day", 0, 23, 12)
        
        # Create prediction input
        pred_year = pred_date.year
        pred_month = pred_date.month
        pred_day = pred_date.day
        pred_dayofweek = pred_date.weekday()
        
        prediction_input = pd.DataFrame({
            'latitude': [pred_lat],
            'longitude': [pred_long],
            'depth': [pred_depth],
            'year': [pred_year],
            'month': [pred_month],
            'day': [pred_day],
            'hour': [pred_hour],
            'dayofweek': [pred_dayofweek]
        })
        
        # Make prediction
        predicted_magnitude = model.predict(prediction_input)[0]
        
        # Calculate estimated energy release
        predicted_energy = 10 ** (1.5 * predicted_magnitude + 4.8)
        predicted_tnt = predicted_energy / (4.184 * 10**9)
        
        # Display prediction with confidence interval
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Magnitude", f"{predicted_magnitude:.2f}", help="Estimated earthquake magnitude on the Richter scale")
        with col2:
            st.metric("Estimated Energy", f"{predicted_tnt:.2f} tons TNT", help="Equivalent to this many tons of TNT explosive")
            
        # Simple damage estimation
        damage_level = ""
        if predicted_magnitude < 3.0:
            damage_level = "Minimal to none - Might be felt by some people, but unlikely to cause damage."
        elif predicted_magnitude < 4.0:
            damage_level = "Minor - May cause shaking of indoor objects but rarely causes damage."
        elif predicted_magnitude < 5.0:
            damage_level = "Light - Felt by most people, minor damage possible to older buildings."
        elif predicted_magnitude < 6.0:
            damage_level = "Moderate - Can cause damage to poorly constructed buildings but limited damage elsewhere."
        elif predicted_magnitude < 7.0:
            damage_level = "Strong - Can be destructive in areas around epicenter."
        else:
            damage_level = "Major to Severe - Can cause serious damage over large areas."
            
        st.info(f"**Potential Impact**: {damage_level}")
        
        # Warning about predictions
        st.warning("""
        ⚠️ **Important Disclaimer**: This prediction is based solely on historical patterns in the selected data.
        It is not a definitive forecast. Scientific earthquake prediction remains challenging, and this tool should 
        not be used for actual hazard assessment or emergency planning. Always refer to official geological survey 
        sources for reliable information.
        """)
        
        # Regional forecast explanation
        st.subheader("Regional Forecast Explanation")
        
        # Find similar historical earthquakes for comparison
        if df.shape[0] > 0:
            # Calculate distance to the prediction point
            df['dist_to_point'] = np.sqrt(
                (df['latitude'] - pred_lat)**2 + 
                (df['longitude'] - pred_long)**2
            )
            
            # Get nearby earthquakes
            nearby = df[df['dist_to_point'] < 5].copy()  # Within ~500km
            
            if len(nearby) > 0:
                st.markdown(f"""
                **Historical Context:** 
                
                In the area near your selected coordinates ({pred_lat:.2f}, {pred_long:.2f}), there have been:
                - {len(nearby)} earthquakes in our dataset
                - Average magnitude: {nearby['mag'].mean():.2f}
                - Maximum magnitude: {nearby['mag'].max():.2f}
                
                The area appears to be {"highly active" if len(nearby) > 10 else "moderately active" if len(nearby) > 3 else "less active"} 
                based on the data available.
                """)
            else:
                st.markdown("""
                **Historical Context:** 
                
                There are no recorded earthquakes near your selected location in our current dataset.
                This could mean:
                1. The area has low seismic activity
                2. We don't have sufficient data for this region
                3. You've selected a location outside the region covered by your data
                """)
        
    else:
        st.warning("Not enough data for reliable prediction modeling. Try expanding your date range or adjusting magnitude filters.")
        st.info("Suggestion: Try selecting a wider date range or lowering the minimum magnitude to include more earthquakes.")

# Tab 4: Risk Analysis
with tab4:
    st.subheader("Earthquake Risk Analysis")
    
    with st.expander("ℹ️ Understanding Earthquake Risk"):
        st.markdown("""
        **What is Earthquake Risk?**
        
        Earthquake risk combines:
        1. **Hazard** - The likelihood of earthquakes occurring
        2. **Exposure** - Population and infrastructure in the area
        3. **Vulnerability** - How well structures can withstand shaking
        
        This section provides a simplified educational view of risk factors based on the data available.
        For official risk assessments, consult your local geological survey or emergency management agencies.
        """)
    
    # Calculate risk metrics
    st.markdown("### Hazard Assessment")
    
    # Create grid for heatmap
    if df.shape[0] > 10:
        # Create a grid of lat/long bins
        lat_bins = np.linspace(df['latitude'].min(), df['latitude'].max(), 20)
        lon_bins = np.linspace(df['longitude'].min(), df['longitude'].max(), 20)
        
        # Assign earthquakes to grid cells
        df['lat_bin'] = pd.cut(df['latitude'], lat_bins, labels=lat_bins[:-1])
        df['lon_bin'] = pd.cut(df['longitude'], lon_bins, labels=lon_bins[:-1])
        
        # Group by grid cell
        grid_data = df.groupby(['lat_bin', 'lon_bin']).agg(
            count=('mag', 'count'),
            avg_mag=('mag', 'mean'),
            max_mag=('mag', 'max'),
            avg_depth=('depth', 'mean')
        ).reset_index()
        
        # Calculate a simple risk score (just for educational purposes)
        # Higher score = more frequent and/or stronger earthquakes
        grid_data['hazard_score'] = (grid_data['count'] * grid_data['avg_mag']) / grid_data['count'].max()
        
        # Plotting the hazard map
        fig_risk = px.density_mapbox(
            grid_data,
            lat='lat_bin',
            lon='lon_bin',
            z='hazard_score',
            radius=30,
            center=dict(lat=0, lon=0),
            zoom=1,
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis",
            labels={"hazard_score": "Hazard Level"},
            title="Relative Earthquake Hazard Levels"
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk factors explanation
        st.markdown("### Key Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Magnitude distribution
            fig_mag_risk = px.pie(
                df, 
                names='magnitude_category',
                title="Earthquake Severity Distribution",
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_mag_risk, use_container_width=True)
        
        with col2:
            # Depth distribution (risk factor)
            df['depth_category'] = pd.cut(
                df['depth'],
                bins=[0, 70, 300, 700],
                labels=['Shallow (<70km)', 'Intermediate (70-300km)', 'Deep (>300km)']
            )
            
            fig_depth_risk = px.pie(
                df, 
                names='depth_category',
                title="Earthquake Depth Distribution",
                color_discrete_sequence=px.colors.sequential.Viridis_r
            )
            st.plotly_chart(fig_depth_risk, use_container_width=True)
        
        # Tsunami potential analysis
        st.subheader("Tsunami Potential Analysis")
        
        tsunami_risk = df['tsunami_potential'].sum()
        tsunami_percent = (tsunami_risk / df.shape[0]) * 100
        
        st.markdown(f"""
        Based on general heuristics (coastal earthquakes > 6.5 magnitude), approximately 
        **{tsunami_risk}** earthquakes ({tsunami_percent:.1f}%) in this dataset had potential tsunami-generating characteristics.
        
        **Remember**: Actual tsunami generation depends on many factors beyond just magnitude, including:
        - Exact earthquake mechanism and direction
        - Ocean depth at the epicenter
        - Coastal topography
        """)
        
        # Risk mitigation advice
        st.subheader("Safety & Preparedness Information")
        
        st.markdown("""
        ### Earthquake Safety Tips
        
        **Before an Earthquake:**
        - Create an emergency plan with your family
        - Prepare an emergency kit with water, food, and supplies
        - Secure heavy furniture to walls
        - Know how to turn off gas, water, and electricity
        
        **During an Earthquake:**
        - DROP to the ground
        - COVER by getting under sturdy furniture
        - HOLD ON until the shaking stops
        - Stay away from windows and exterior walls
        
        **After an Earthquake:**
        - Check for injuries and provide first aid
        - Look for fire hazards or gas leaks
        - Listen for emergency information
        - Be prepared for aftershocks
        
        **For official guidance, visit:**
        - [U.S. Geological Survey](https://www.usgs.gov/natural-hazards/earthquake-hazards)
        - [Ready.gov Earthquake Preparedness](https://www.ready.gov/earthquakes)
        """)
        
    else:
        st.warning("Not enough data for risk analysis. Try expanding your data range or region.")

# Tab 5: Mobile Alert System
with tab5:
    st.subheader("Early Warning Alert Simulation")
    
    with st.expander("ℹ️ About Earthquake Early Warning Systems"):
        st.markdown("""
        **How Do Early Warning Systems Work?**
        
        Real earthquake early warning systems can provide seconds to minutes of advance warning before shaking arrives:
        
        1. Seismic sensors detect initial earthquake waves (P-waves)
        2. Computers quickly analyze the data and estimate location and magnitude
        3. Alerts are sent before the more damaging waves (S-waves) arrive
        4. The warning time depends on your distance from the epicenter
        
        This simulation demonstrates the concept of early warning. In a real system, alerts would be sent automatically based on sensor data.
        
        **Note**: Many regions now have official early warning apps available for mobile devices.
        """)
    
    st.markdown("""
    This simulation shows how an early warning system might work. Select an earthquake from the data to simulate
    sending alerts to different locations based on their distance from the epicenter.
    """)
    
    # Get significant earthquakes for the simulation
    if df.shape[0] > 0:
        # Filter to meaningful earthquakes for demonstration
        sig_quakes = df[df['mag'] >= 5.0].copy() if df['mag'].max() >= 5.0 else df.nlargest(5, 'mag')
        
        if not sig_quakes.empty:
            # Format for selection
            sig_quakes['description'] = sig_quakes.apply(
                lambda row: f"{row['place']} - M{row['mag']} ({row['time'].strftime('%Y-%m-%d')})", 
                axis=1
            )
            
            selected_quake = st.selectbox(
                "Select an earthquake for the simulation:",
                options=sig_quakes['description'].tolist()
            )
            
            # Get the selected earthquake data
            quake_data = sig_quakes[sig_quakes['description'] == selected_quake].iloc[0]
            
            # Display earthquake info
            st.markdown(f"""
            ### Selected Event: {quake_data['place']}
            - **Magnitude**: {quake_data['mag']}
            - **Depth**: {quake_data['depth']} km
            - **Date/Time**: {quake_data['time']}
            - **Location**: {quake_data['latitude']:.4f}°, {quake_data['longitude']:.4f}°
            """)
            
            # Define some sample cities/locations for the simulation
            # Calculate their distances from the epicenter
            sample_locations = [
                {"name": "City A", "lat": quake_data['latitude'] + 0.5, "lon": quake_data['longitude'] + 0.5},
                {"name": "City B", "lat": quake_data['latitude'] + 1.0, "lon": quake_data['longitude'] + 1.0},
                {"name": "City C", "lat": quake_data['latitude'] + 2.0, "lon": quake_data['longitude'] + 2.0},
                {"name": "City D", "lat": quake_data['latitude'] + 3.0, "lon": quake_data['longitude'] + 3.0},
                {"name": "City E", "lat": quake_data['latitude'] - 1.0, "lon": quake_data['longitude'] - 1.0},
            ]
            
            # Calculate distances and warning times
            for location in sample_locations:
                # Simple distance calculation (approximation)
                # In km, using 111 km per degree of lat/lon (approximation)
                dx = (location['lon'] - quake_data['longitude']) * 111 * math.cos(math.radians(location['lat']))
                dy = (location['lat'] - quake_data['latitude']) * 111
                distance = math.sqrt(dx*dx + dy*dy)
                
                # P-wave speed ~6 km/s, S-wave ~3.5 km/s
                p_wave_time = distance / 6.0  # seconds
                s_wave_time = distance / 3.5  # seconds
                warning_time = s_wave_time - p_wave_time  # seconds
                
                location['distance'] = distance
                location['warning_time'] = warning_time
            
            # Create a map for the simulation
            alert_map = folium.Map(
                location=[quake_data['latitude'], quake_data['longitude']],
                zoom_start=6
            )
            
            # Add the earthquake epicenter
            folium.Marker(
                [quake_data['latitude'], quake_data['longitude']],
                popup=f"Epicenter: M{quake_data['mag']}",
                icon=folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
            ).add_to(alert_map)
            
            # Add circles for P-wave and S-wave propagation
            folium.Circle(
                radius=50000,  # 50 km
                location=[quake_data['latitude'], quake_data['longitude']],
                popup='P-wave (50km)',
                color='yellow',
                fill=True
            ).add_to(alert_map)
            
            folium.Circle(
                radius=100000,  # 100 km
                location=[quake_data['latitude'], quake_data['longitude']],
                popup='P-wave (100km)',
                color='orange',
                fill=True
            ).add_to(alert_map)
            
            folium.Circle(
                radius=50000 * 3.5/6.0,  # S-wave when P-wave at 50km
                location=[quake_data['latitude'], quake_data['longitude']],
                popup='S-wave',
                color='red',
                fill=True
            ).add_to(alert_map)
            
            # Add the cities to the map
            for location in sample_locations:
                folium.Marker(
                    [location['lat'], location['lon']],
                    popup=f"{location['name']}: {location['warning_time']:.1f}s warning",
                    icon=folium.Icon(color='blue' if location['warning_time'] > 0 else 'gray', icon='building', prefix='fa')
                ).add_to(alert_map)
            
            # Display the map
            st.subheader("Early Warning Map")
            folium_static(alert_map)
            
            # Create a simulation of the alert process
            st.subheader("Alert Timeline Simulation")
            
            if st.button("Run Simulation"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate earthquake detection
                status_text.text("Earthquake occurs at epicenter...")
                time.sleep(1)
                progress_bar.progress(10)
                
                status_text.text("P-waves detected by seismic network...")
                time.sleep(1)
                progress_bar.progress(20)
                
                status_text.text("System calculating magnitude and location...")
                time.sleep(1)
                progress_bar.progress(40)
                
                status_text.text(f"Alert issued! Estimated magnitude: {quake_data['mag']:.1f}")
                time.sleep(1)
                progress_bar.progress(60)
                
                # Show alerts for each city
                time.sleep(1)
                progress_bar.progress(80)
                
                # Display the alert results in a table
                alert_results = pd.DataFrame(sample_locations)
                alert_results = alert_results[['name', 'distance', 'warning_time']]
                alert_results.columns = ['Location', 'Distance (km)', 'Warning Time (seconds)']
                alert_results['Distance (km)'] = alert_results['Distance (km)'].round(1)
                alert_results['Warning Time (seconds)'] = alert_results['Warning Time (seconds)'].round(1)
                alert_results['Alert Status'] = alert_results['Warning Time (seconds)'].apply(
                    lambda x: "Alert Delivered ✓" if x > 3 else "Minimal Warning ⚠️" if x > 0 else "No Warning Possible ✗"
                )
                
                status_text.text("Alert delivery complete!")
                progress_bar.progress(100)
                
                st.table(alert_results)
                
                st.markdown("""
                **What this means:**
                - Locations closer to the epicenter get less warning time
                - Even a few seconds can be enough to take protective actions (drop, cover, hold on)
                - Areas far from the epicenter might get tens of seconds of warning
                """)
            
            # Mobile app example
            st.subheader("Example Mobile Alert")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Create a simple mock-up of a phone alert
                alert_html = f"""
                <div style="border: 2px solid #ff4444; border-radius: 10px; padding: 15px; background-color: #fff0f0; max-width: 300px; font-family: Arial;">
                    <div style="display: flex; align-items: center;">
                        <div style="color: red; font-size: 24px; margin-right: 10px;">⚠️</div>
                        <div style="color: red; font-weight: bold; font-size: 18px;">EARTHQUAKE ALERT</div>
                    </div>
                    <div style="margin-top: 10px;">
                        <div style="font-weight: bold;">Magnitude {quake_data['mag']:.1f} Earthquake Detected</div>
                        <div style="font-size: 14px; margin-top: 5px;">Location: {quake_data['place']}</div>
                        <div style="font-size: 24px; font-weight: bold; text-align: center; margin: 15px 0; color: red;">
                            DROP! COVER! HOLD ON!
                        </div>
                        <div style="font-size: 12px;">Expect shaking shortly. Take protective actions now.</div>
                    </div>
                </div>
                """
                st.markdown(alert_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                **How a Real Alert Works:**
                
                1. When an earthquake is detected, alerts can be sent through:
                   - Mobile apps
                   - Emergency alert systems
                   - Smart devices
                   - Public warning systems
                
                2. The alert provides:
                   - Estimated magnitude
                   - Expected shaking intensity
                   - Seconds until shaking arrives
                   - Protective action guidance
                
                3. This gives people time to:
                   - Drop, Cover, and Hold On
                   - Move away from hazards
                   - Stop vehicles or machinery
                   - Mentally prepare for shaking
                """)
            
            st.info("""
            **Real Systems Available Now:**
            
            Many regions have actual earthquake early warning systems with mobile apps, including:
            - ShakeAlert (US West Coast)
            - Japan Earthquake Early Warning
            - Mexico's SASMEX
            - Taiwan's Earthquake Early Warning System
            
            Check your local geological survey or emergency management agencies to see if early warning is
            available in your area.
            """)
            
        else:
            st.warning("No significant earthquakes found in the dataset for simulation.")
    else:
        st.warning("No earthquake data available for the alert simulation.")

# Tab 6: Data Explorer
with tab6:
    st.subheader("Earthquake Data Explorer")
    
    with st.expander("ℹ️ Help: Exploring the Raw Data"):
        st.markdown("""
        **What can I do here?**
        - View the actual earthquake data
        - Filter by magnitude, depth, or date
        - Search for specific locations
        - Download the data for your own analysis
        
        This is useful for researchers or if you want to look at specific earthquakes in detail.
        """)
    
    # Data filtering options
    st.markdown("### Filter Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        mag_filter = st.slider("Filter by Magnitude", 
                              min_value=float(df['mag'].min()), 
                              max_value=float(df['mag'].max()),
                              value=(float(df['mag'].min()), float(df['mag'].max())))
    
    with col2:
        depth_filter = st.slider("Filter by Depth (km)", 
                                min_value=float(df['depth'].min()), 
                                max_value=float(df['depth'].max()),
                                value=(float(df['depth'].min()), float(df['depth'].max())))
        
    with col3:
        # Location search
        location_search = st.text_input("Search locations (e.g., 'California')")
    
    # Apply filters
    filtered_df = df[(df['mag'] >= mag_filter[0]) & 
                    (df['mag'] <= mag_filter[1]) &
                    (df['depth'] >= depth_filter[0]) &
                    (df['depth'] <= depth_filter[1])]
    
    # Apply location search if provided
    if location_search:
        filtered_df = filtered_df[filtered_df['place'].str.contains(location_search, case=False)]
    
    # Sort options
    sort_col, sort_dir = st.columns(2)
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=["time", "mag", "depth", "place"],
            index=0
        )
    
    with sort_dir:
        sort_ascending = st.radio(
            "Order",
            options=["Descending", "Ascending"],
            index=0,
            horizontal=True
        )
    
    # Apply sorting
    filtered_df = filtered_df.sort_values(
        by=sort_by, 
        ascending=(sort_ascending == "Ascending")
    )
    
    # Show data count and provide download option
    st.markdown(f"### Showing {filtered_df.shape[0]} earthquakes")
    
   # Add summary stats for filtered data
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Magnitude", f"{filtered_df['mag'].mean():.2f}")
    with col2:
        st.metric("Average Depth", f"{filtered_df['depth'].mean():.2f} km")
    with col3:
        st.metric("Total Energy", f"{filtered_df['energy_tnt'].sum():.2f} tons TNT")
    
    # Add time range of filtered data
    if filtered_df.shape[0] > 1:
        st.info(f"Time range: {filtered_df['time'].min().strftime('%Y-%m-%d')} to {filtered_df['time'].max().strftime('%Y-%m-%d')}")
    
    # Add magnitude distribution of filtered data
    mag_count = filtered_df['magnitude_category'].value_counts().reset_index()
    mag_count.columns = ['Category', 'Count']
    
    fig_mag_filtered = px.bar(
        mag_count,
        x='Category',
        y='Count',
        color='Category',
        color_discrete_sequence=px.colors.sequential.Inferno,
        title="Magnitude Distribution of Filtered Earthquakes"
    )
    
    st.plotly_chart(fig_mag_filtered, use_container_width=True)
    
    # Display downloadable CSV
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_earthquake_data.csv",
        mime="text/csv",
    )

    # Display the dataframe with the most relevant columns
    st.dataframe(
        filtered_df[['time', 'place', 'mag', 'depth', 'latitude', 'longitude', 'energy_tnt']],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No earthquakes match your current filters. Try adjusting your criteria.")
