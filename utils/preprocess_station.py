import pandas as pd
import geopandas as gpd


def import_rescue_station(addr, crs='epsg:4326'):
    rescue = pd.read_csv(addr)
    rescue = gpd.GeoDataFrame(
        rescue,
        geometry=gpd.points_from_xy(rescue['lon'], rescue['lat'])).set_crs(crs)
    return rescue


def add_nearest_segment(point, road_segment):
    i = point.distance(road_segment.geometry).sort_values().index[0]
    id = road_segment.iloc[i]['osmid']
    return id


def add_nearest_intersection(point, road_intersection):
    i = point.distance(road_intersection.geometry).sort_values().index[0]
    id = road_intersection.iloc[i]['osmid']
    return id
