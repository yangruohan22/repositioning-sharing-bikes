import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import numpy as np

# Set plotting style for better aesthetics
plt.style.use('ggplot')

# 1. Data Loading and Preprocessing
# ==============================================================================
try:
    df = pd.read_excel('processed_bike_data.xlsx')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'processed_bike_data.xlsx' not found. Please ensure the file is in the same directory.")
    exit()

# Convert time columns to datetime objects
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# 2. Histograms and Descriptive Statistics
# ==============================================================================
print("\nGenerating histograms and calculating statistics...")

# 2.1 Trip Duration Histogram and Statistics
# ------------------------------------------------------------------------------
# Calculate descriptive statistics
duration_mean = df['duration_minutes'].mean()
duration_median = df['duration_minutes'].median()
duration_std = df['duration_minutes'].std()

print(f"\nTrip Duration (minutes) Statistics:")
print(f"  - Mean: {duration_mean:.2f}")
print(f"  - Median: {duration_median:.2f}")
print(f"  - Standard Deviation: {duration_std:.2f}")

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['duration_minutes'], bins=50, kde=True)
plt.title('Trip Duration Distribution', fontsize=16)
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Trip Count', fontsize=12)
plt.tight_layout()
plt.savefig('pictures/trip_duration_histogram.png')
plt.close()

# 2.2 Trip Distance Histogram and Statistics
# ------------------------------------------------------------------------------
# Calculate descriptive statistics
distance_mean = df['trip_distance_km'].mean()
distance_median = df['trip_distance_km'].median()
distance_std = df['trip_distance_km'].std()

print(f"\nTrip Distance (km) Statistics:")
print(f"  - Mean: {distance_mean:.2f}")
print(f"  - Median: {distance_median:.2f}")
print(f"  - Standard Deviation: {distance_std:.2f}")

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['trip_distance_km'], bins=50, kde=True)
plt.title('Trip Distance Distribution', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Trip Count', fontsize=12)
plt.tight_layout()
plt.savefig('pictures/trip_distance_histogram.png')
plt.close()

# 2.3 Hourly Trip Count Histogram (30-minute intervals)
# ------------------------------------------------------------------------------
# Calculate descriptive statistics
hourly_trip_counts = df['start_time'].dt.floor('30min').value_counts()
hourly_mean = hourly_trip_counts.mean()
hourly_median = hourly_trip_counts.median()
hourly_std = hourly_trip_counts.std()

print(f"\nHourly Trip Count Statistics (30-min intervals):")
print(f"  - Mean: {hourly_mean:.2f}")
print(f"  - Median: {hourly_median:.2f}")
print(f"  - Standard Deviation: {hourly_std:.2f}")

# Plot histogram
df['start_minute_of_day'] = df['start_time'].dt.hour * 60 + df['start_time'].dt.minute
bins = np.arange(0, 24 * 60 + 30, 30)
df['time_bin'] = pd.cut(df['start_minute_of_day'], bins=bins, right=False)
order_count_by_time_bin = df['time_bin'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
order_count_by_time_bin.plot(kind='bar', color='skyblue')

tick_positions = np.arange(0, len(order_count_by_time_bin), 2)
tick_labels = [f'{h:02d}:00' for h in range(24)]
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.title('Trip Count Distribution per Time of Day', fontsize=16)
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Trip Count', fontsize=12)
plt.tight_layout()
plt.savefig('pictures/hourly_trip_count.png')
plt.close()

# 2.4 Daily Bike Usage Histogram and Statistics
# ------------------------------------------------------------------------------
df['start_date'] = df['start_time'].dt.date
bike_usage_per_day = df.groupby(['bike_id', 'start_date']).size().reset_index(name='daily_rides')
daily_rides_counts = bike_usage_per_day['daily_rides'].value_counts().sort_index()

# Calculate descriptive statistics
usage_mean = bike_usage_per_day['daily_rides'].mean()
usage_median = bike_usage_per_day['daily_rides'].median()
usage_std = bike_usage_per_day['daily_rides'].std()

print(f"\nDaily Bike Usage Statistics:")
print(f"  - Mean: {usage_mean:.2f}")
print(f"  - Median: {usage_median:.2f}")
print(f"  - Standard Deviation: {usage_std:.2f}")

# Plot histogram
plt.figure(figsize=(10, 6))
daily_rides_counts.plot(kind='bar', color='salmon')

plt.title('Daily Bike Usage Frequency', fontsize=16)
plt.xlabel('Number of Trips Per Day', fontsize=12)
plt.ylabel('Bike-Days (Number of bikes used X times)', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('pictures/daily_bike_usage.png')
plt.close()

# 3. Heatmaps and Scatter Plots
# ==============================================================================
print("\nGenerating correlation heatmap and pair plot...")

# 3.1 Variable Correlation Heatmap and Pair Plot Matrix
# ------------------------------------------------------------------------------

numerical_vars = ['start_time', 'trip_distance_km', 'duration_minutes', 'start_lat', 'start_lon']
corr_matrix = df[numerical_vars].corr()

# Plotting the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, vmin=-1, vmax=1)
plt.title('Variable Correlation Heatmap', fontsize=16)
plt.savefig('pictures/variable_correlation.png')
plt.close()

# Plotting the pair plot matrix
sns.pairplot(df[numerical_vars], diag_kind='kde')
plt.suptitle('Variable Pair Plot Matrix', y=1.02, fontsize=16)
plt.savefig('pair_plot.png')
plt.close()

# 3.2 Peak Hour Location Heatmaps (using Folium)
# ------------------------------------------------------------------------------
print("\nGenerating separate start location heatmaps for each peak hour...")

# Define peak hours
peaks = {
    'morning': {'start': 7, 'end': 9, 'end_minute': 30, 'start_minute': 30},
    'lunch': {'start': 11, 'end': 13, 'end_minute': 0, 'start_minute': 30},
    'evening': {'start': 17, 'end': 18, 'end_minute': 30, 'start_minute': 0}
}

for peak_name, peak_times in peaks.items():
    is_peak = ((df['start_time'].dt.hour == peak_times['start']) & (
                df['start_time'].dt.minute >= peak_times['start_minute'])) | \
              ((df['start_time'].dt.hour > peak_times['start']) & (df['start_time'].dt.hour < peak_times['end'])) | \
              ((df['start_time'].dt.hour == peak_times['end']) & (
                          df['start_time'].dt.minute <= peak_times['end_minute']))

    peak_df = df[is_peak]

    if peak_df.empty:
        print(f"Warning: No data found for {peak_name} peak. Heatmap will not be generated.")
        continue

    # Get start location data for the heatmap
    start_locations = peak_df[['start_lat', 'start_lon']].values.tolist()

    # Create a base map centered on the average coordinates
    map_center = [peak_df['start_lat'].mean(), peak_df['start_lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', attr='高德地图')

    # Add the heatmap with adjusted parameters
    # radius: Controls the size of the individual heat points. Smaller for higher detail.
    # blur: Controls the smoothness of the heatmap. Smaller for sharper distinction.
    # The default for both is 15. We can try 10 or 12 for better distinction.
    HeatMap(start_locations, name=f'{peak_name.capitalize()} Peak Start Locations', radius=12, blur=12).add_to(m)

    # Save the map to a file
    filename = f'pictures/{peak_name}_peak_heatmap.html'
    m.save(filename)
    print(f"Start location heatmap for {peak_name} peak saved as '{filename}'.")

print("\nAll visualization and analysis tasks are complete.")