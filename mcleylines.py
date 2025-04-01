import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from sklearn.linear_model import RANSACRegressor
import random
import requests
import io
from scipy.stats import chi2_contingency, fisher_exact
from itertools import combinations
from geopandas.tools import sjoin

def get_mcdonalds_locations():
    population_centers = [
        # (lat, lon, weight) - weight determines how many McDonald's are clustered here
        (40.7128, -74.0060, 15),  # New York
        (34.0522, -118.2437, 15),  # Los Angeles
        (51.5074, -0.1278, 10),    # London
        (48.8566, 2.3522, 10),     # Paris
        (35.6762, 139.6503, 12),   # Tokyo
        (22.3193, 114.1694, 8),    # Hong Kong
        (19.4326, -99.1332, 8),    # Mexico City
        (-33.8688, 151.2093, 7),   # Sydney
        (55.7558, 37.6173, 7),     # Moscow
        (-23.5505, -46.6333, 8),   # São Paulo
    ]
    data = []
    id_counter = 1
    
    for lat, lon, weight in population_centers:
        for _ in range(weight):
            noise_lat = np.random.normal(-1, 1)
            noise_lon = np.random.normal(-1, 1)
            data.append({
                'id': f'MD_{id_counter}',
                'lat': lat + noise_lat,
                'lon': lon + noise_lon
            })
            id_counter += 1
    return pd.DataFrame(data)

def find_linear_mcdonalds(mcdonalds_gdf, min_points=50, d_threshold=0.1):
    best_lines = []
    coords = np.array([(point.x, point.y) for point in mcdonalds_gdf.geometry])
    angle_step = np.pi / 8
    angle_bins = np.arange(0, np.pi*2, angle_step)
    
    for angle_idx, angle in enumerate(angle_bins):
        # print(f"Searching For McLeyLine Alignment in {angle}")
        cos_theta, sin_theta = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
        rotated_coords = np.dot(coords, rotation_matrix)
        sorted_indices = np.argsort(rotated_coords[:, 0])
        sorted_coords = rotated_coords[sorted_indices]
        
        lines_in_direction = []
        for _ in range(10):
            sample_indices = np.random.choice(len(coords), size=int(len(coords) * 0.8), replace=False)
            sample_points = coords[sample_indices]

            if len(sample_points) < min_points:
                continue
            
            X = sample_points[:, 0].reshape(-1, 1)  # Longitude
            y = sample_points[:, 1]  # Latitude

            ransac = RANSACRegressor(min_samples=min_points, 
                                    residual_threshold=d_threshold,
                                    random_state=np.random.randint(0, 50))
            


            ransac.fit(X,y)
            inlier_mask = ransac.inlier_mask_
            inlier_indices = np.where(inlier_mask)[0]

            if len(inlier_indices) >= min_points:
                original_indices = sorted_indices[inlier_indices]
                aligned_mcdonalds = mcdonalds_gdf.iloc[original_indices].copy()
                
                min_x, max_x = X[inlier_indices].min(), X[inlier_indices].max()

                inlier_points_transformed = X[inlier_indices]
                line = LineString(line_points_transformed)
                                

                lines_in_direction.append({
                    'line': line,
                    'points': aligned_mcdonalds,
                    'count': len(aligned_mcdonalds),

                })
            
        best_lines.extend(sorted(lines_in_direction, key=lambda x: x['count'], reverse=True))


    return best_lines
def visualise():
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    mcdonalds_df = get_mcdonalds_locations()
    mcdonalds_csv = pd.read_csv('McDonalds.csv')
    geometry = [Point(xy) for xy in zip(mcdonalds_csv['longitude'], mcdonalds_csv['latitude'])]
    
    
    mcdonalds_gdf = gpd.GeoDataFrame(mcdonalds_csv, geometry=geometry, crs="EPSG:4326")
    


    world = gpd.read_file(url)
    # print(world)
    uk =  world[world.NAME == "United Kingdom"]
    # print(uk)
    world = uk
    uk_geometry = uk.geometry.unary_union

    uk_mcdonalds_mask = mcdonalds_gdf.within(uk_geometry)
    uk_mcdonalds = mcdonalds_gdf[uk_mcdonalds_mask]
    mcdonalds_gdf = uk_mcdonalds
    lines = find_linear_mcdonalds(mcdonalds_gdf)
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, line_info in enumerate(lines[:250]):
        line_gdf = gpd.GeoDataFrame(geometry=[line_info['line']], crs='EPSG:4326')
        line_gdf.plot(ax=ax, color='red', linewidth=0.5, 
                      label=f'Lay Line {i+1}')
        line_info['points'].plot(ax=ax, color='blue', markersize=0.1, alpha=0.5)
    
    world.plot(ax=ax, color='lightgray')
    mcdonalds_gdf.plot(ax=ax)

    plt.tight_layout()
    plt.savefig('mcdonalds_laylines_2.png', dpi=500)
    
if __name__ == "__main__":
    visualise()
