import pandas as pd
import geopandas as gpd


def import_rescue_station(addr: str, crs='epsg:4326'):
    """
    Import rescue station data from a CSV file and convert it to a GeoDataFrame.

    Args:
        addr (str): Path to the CSV file containing rescue station data with 'lon' and 'lat' columns.
        crs (str, optional): Coordinate reference system to use. Defaults to 'epsg:4326'.

    Returns:
        GeoDataFrame: Rescue stations as a GeoDataFrame with point geometries.
    """
    rescue = pd.read_csv(addr)
    rescue = gpd.GeoDataFrame(
        rescue,
        geometry=gpd.points_from_xy(rescue['lon'], rescue['lat'])).set_crs(crs)
    return rescue


def add_nearest_segment(point, road_segment):
    """
    Find the nearest road segment to a given point.

    Args:
        point (shapely.geometry.Point): The point to search from.
        road_segment (GeoDataFrame): GeoDataFrame of road segments with 'osmid' column.

    Returns:
        id: The 'osmid' of the nearest road segment.
    """
    i = point.distance(road_segment.geometry).sort_values().index[0]
    id = road_segment.iloc[i]['osmid']
    return id


def add_nearest_intersection(point, road_intersection):
    """
    Find the nearest road intersection to a given point.

    Args:
        point (shapely.geometry.Point): The point to search from.
        road_intersection (GeoDataFrame): GeoDataFrame of intersections with 'osmid' column.

    Returns:
        id: The 'osmid' of the nearest intersection.
    """
    i = point.distance(road_intersection.geometry).sort_values().index[0]
    id = road_intersection.iloc[i]['osmid']
    return id


def check_closure_n_occupation(row, station_col, time_col, incidents):
    """
    Check if a rescue station is closed or occupied at a given time based on incident history.

    Args:
        row (Series): Row containing station and time information.
        station_col (str): Column name for the station identifier in the row.
        time_col (str): Column name for the time in the row.
        incidents (DataFrame): DataFrame of incidents with 'Rescue Squad Number' and 'Call Date and Time'.

    Returns:
        tuple: (is_closed (bool or None), is_occupied (bool or None))
    """
    station = row[station_col]
    time = row[time_col]

    station_incidents = incidents[incidents['Rescue Squad Number'] == station]
    station_incidents = station_incidents[station_incidents['Call Date and Time'] <= time]
    station_incidents = station_incidents.sort_values(by='Call Date and Time', ascending=False)
    if len(station_incidents) == 0:
        return None, None
    last_incident_ends = station_incidents.iloc[0]['Close Date and Time']

    t_delta_all = incidents['Call Date and Time'].diff()
    t_delta_all = t_delta_all[t_delta_all <= t_delta_all.quantile(0.9)]
    t_delta = time - last_incident_ends
    t_delta_z_score = (t_delta - t_delta_all.mean()) / t_delta_all.std()
    threshold_95 = 1.65

    return time <= last_incident_ends, t_delta_z_score >= threshold_95
