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


def check_occupation(row, station_col, time_col, incidents):
    station = row[station_col]
    time = row[time_col]

    station_incidents = incidents[incidents['Rescue Squad Number'] == station]
    station_incidents = station_incidents[station_incidents['Call Date and Time'] <= time]
    station_incidents = station_incidents.sort_values(by='Call Date and Time', ascending=False)
    last_incident_ends = station_incidents.iloc[0]['Close Date and Time']

    t_delta_all = incidents['Call Date and Time'].diff()
    t_delta_all = t_delta_all[t_delta_all <= t_delta_all.quantile(0.9)]
    t_delta = time - last_incident_ends
    t_delta_z_score = (t_delta - t_delta_all.mean()) / t_delta_all.std()
    threshold_95 = 1.65

    return time <= last_incident_ends, t_delta_z_score >= threshold_95
