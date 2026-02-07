from networkx import MultiDiGraph
import osmnx as ox
from tqdm import tqdm
import time
import os
from typing import cast
from tqdm import tqdm
import map_poster.cli
from map_poster.caching import cache_get, cache_set, CacheError
from shapely.geometry import box
import geopandas as gpd
from geopandas import GeoDataFrame
from pathlib import Path
from geopy.geocoders import Nominatim
import asyncio
import numpy as np

WATER_POLY_DIR = Path("cache/water_polygons")

def fetch_ocean_polygons(point, dist, refresh_cache, coastline=None):
    """
    Fetch preprocessed ocean polygons for a bounding box around `point`.
    Only fetches if coastline data is present.
    Uses caching.
    """
    # Skip if no coastline present
    if coastline is None or coastline.empty:
        return gpd.GeoDataFrame()  # return empty GeoDataFrame

    lat, lon = point
    lat_offset = dist / 111_000  # 1 deg latitude ≈ 111 km
    lon_offset = dist / (111_000 * np.cos(np.radians(lat)))  # scale by latitude

    # Compute bounding box from coastline
    minx, miny, maxx, maxy = coastline.total_bounds
    minx, miny = min(minx, lon - lon_offset), min(miny, lat - lat_offset)
    maxx, maxy = max(maxx, lon + lon_offset), max(maxy, lat + lat_offset)

    cache_key = f"ocean_{minx}_{miny}_{maxx}_{maxy}"
    cached = cache_get(cache_key)
    if cached is not None and not refresh_cache:
        tqdm.write("✓ Using cached ocean polygons")
        return cast(GeoDataFrame, cached)
    
    ocean_path = map_poster.cli.ensure_water_polygons(WATER_POLY_DIR)
    if not os.path.exists(ocean_path):
        raise FileNotFoundError(
            f"Water polygons shapefile not found: {ocean_path}\n"
            "Download from: https://osmdata.openstreetmap.de/data/water-polygons.html"
        )

    tqdm.write("Loading ocean polygons...")
    ocean_gdf = gpd.read_file(ocean_path)

    # Clip to bounding box
    bbox_geom = box(minx, miny, maxx, maxy)
    clipped_ocean = gpd.clip(ocean_gdf, bbox_geom)

    try:
        cache_set(cache_key, clipped_ocean)
    except CacheError as e:
        tqdm.write(e)

    return clipped_ocean    


def fetch_graph(point, dist, refresh_cache) -> MultiDiGraph | None:
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None and refresh_cache is False:
        tqdm.write("✓ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True, retain_all=True)
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, G)
        except CacheError as e:
            tqdm.write(e)
        return G
    except Exception as e:
        tqdm.write(f"OSMnx error while fetching graph: {e}")
        return None

def fetch_features(point, dist, refresh_cache, tags, name) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None and refresh_cache is False:
        tqdm.write(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            tqdm.write(e)
        return data
    except Exception as e:
        if "No matching features" not in str(e):
            tqdm.write(f"⚠️ OSMnx error while fetching {name}: {e}")
        return None
    
def get_coordinates(city, country, refresh_cache):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached and not refresh_cache:
        print(f"✓ Using cached coordinates for {city}, {country}")
        print(f"✓ Coordinates: {cached}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}")

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError("Geocoder returned a coroutine while an event loop is already running. Run this script in a synchronous environment.")
            location = loop.run_until_complete(location)
    
    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")