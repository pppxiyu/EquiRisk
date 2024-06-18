import pandas as pd
import geopandas as gpd


def geocode_incident(data_addr, time_range, save_addr):
    import geopy as gpy
    from geopy.extra.rate_limiter import RateLimiter

    locator = gpy.geocoders.ArcGIS()
    geocode = RateLimiter(locator.geocode, min_delay_seconds=0.1)

    data = pd.read_csv(data_addr)
    data['Call Date and Time'] = pd.to_datetime(data['Call Date and Time'], format="%Y-%m-%d %H:%M:%S")
    data = data[(data['Call Date and Time'] >= time_range[0]) & (data['Call Date and Time'] <= time_range[1])]
    data['Country'] = ['USA'] * len(data)
    data['Address'] = data['Block Address'].str.cat([
        pd.Series(', ', index=data.index),
        data['City'],
        pd.Series(', ', index=data.index),
        data['State'],
        pd.Series(', ', index=data.index),
        data['Country']
    ], join="left")

    data['IncidentFullInfo'] = data['Address'].apply(geocode)
    data['IncidentCoor'] = data['IncidentFullInfo'].apply(
        lambda loc: tuple(loc.point) if loc else None
    )
    data['IncidentFullInfo'] = data['IncidentFullInfo'].astype(str)
    data[['IncidentLat', 'IncidentLon', 'IncidentElevation']] = pd.DataFrame(
        data['IncidentCoor'].tolist(), index=data.index,
    )

    data.to_csv(save_addr, index=False)
    return


def import_incident(address):
    from shapely.geometry import box

    data = pd.read_csv(address)

    time_cols = [
        'Call Date and Time', 'Entry Date and Time', 'Dispatch Date and Time',
        'En route Date and Time', 'On Scene Date and Time', 'Close Date and Time',
    ]
    for col in time_cols:
        data[col] = pd.to_datetime(data[col], format="%Y-%m-%d %H:%M:%S")
        data[col] = data[col].dt.tz_localize('America/New_York')
        data[col] = data[col].dt.tz_convert('UTC')

    data['DispatchTime'] = (data['Dispatch Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['EnRouteTime'] = (data['En route Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['TravelTime'] = (data['On Scene Date and Time'] - data['En route Date and Time']).astype("timedelta64[s]")
    data['ResponseTime'] = (data['En route Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['HourInDay'] = data['Call Date and Time'].dt.hour
    data['DayOfWeek'] = data['Call Date and Time'].dt.dayofweek

    data = data.set_index('Call Date and Time')
    data = data.sort_index()
    data = data.reset_index()

    if 'IncidentPoint' not in data.columns:
        data['IncidentPoint'] = gpd.GeoSeries(gpd.points_from_xy(
            y=data['IncidentLat'], x=data['IncidentLon'],
        ), index=data.index, crs="EPSG:4326")
        data = gpd.GeoDataFrame(data, geometry=data['IncidentPoint'])

    minx, miny, maxx, maxy = [-76.227826, 36.550525, 75.867555, 36.931973]
    bbox = box(minx, miny, maxx, maxy)
    data = data[data.intersects(bbox)]

    data['incident_id'] = list(range(1, len(data) + 1))

    return data


def add_actual_rescue_station(incident, rescue):
    rescue = rescue.drop('geometry', axis=1)
    incident = incident[incident['Rescue Squad Number'].isin(rescue.Number.to_list())]
    incident = incident.merge(rescue, how='left', left_on='Rescue Squad Number', right_on='Number')
    return incident


def add_nearest_rescue_station(incident, rescue):
    original_crs = incident.crs
    incident = incident.to_crs('epsg:32633')
    rescue = rescue.to_crs('epsg:32633')
    assert rescue.crs == incident.crs
    incident = incident.sjoin_nearest(rescue, how='left', lsuffix='actual', rsuffix='nearest')
    incident = incident.to_crs(original_crs)
    return incident


def add_period_label(incident, label_dict):
    from datetime import datetime, timedelta

    incident['period_label'] = len(incident) * ['']

    assert len(list(label_dict.items())) == 2, 'Only include the begin and end labels'
    start = list(label_dict.items())[0]
    end = list(label_dict.items())[1]

    start_time = datetime.strptime(start[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end[0], '%Y-%m-%d %H:%M:%S')
    time_intervals = pd.date_range(
        start=start_time, end=end_time + timedelta(hours=1), freq='H'
    ).strftime('%Y-%m-%d %H:%M:%S').tolist()

    start_label = start[1]
    end_label = end[1]
    labels = list(range(start_label, end_label + 1))

    assert len(time_intervals) - 1 == len(labels)
    for i, l in zip(range(len(time_intervals) - 1), labels):
        t_1 = time_intervals[i]
        t_2 = time_intervals[i + 1]
        incident.loc[
            (incident['Call Date and Time'] >= t_1) & (incident['Call Date and Time'] <= t_2),
            'period_label'] = l

    return incident


def convert_feature_class_to_df(incident, feature_class_addr, label_list):
    import arcpy
    import warnings

    df_list = []
    for l in label_list:
        if arcpy.Exists(f'{feature_class_addr}_{l}') is not True:
            warnings.warn(f'route_results_{l} is missing')
            continue

        arr = arcpy.da.FeatureClassToNumPyArray(
            f'{feature_class_addr}_{l}', ('Name', 'Total_Seconds')
        )
        df = pd.DataFrame(arr)
        df['rescue_name'] = df['Name'].str.split('-').str[0]
        df['incident_id'] = df['Name'].str.split('-').str[1].astype('int64')
        df_list.append(df)

    df = pd.concat(df_list, axis=0)
    incident = incident.merge(df[['incident_id', 'Total_Seconds']], how='left', on='incident_id')

    return incident


def add_inaccessible_routes(incidents, addr_inaccessible_routes, fill_value=-999):
    import json
    import numpy as np

    with open(addr_inaccessible_routes, 'r') as f:
        inaccessible_routes = json.load(f)
    flood_period = inaccessible_routes['flood']
    # Normal time inaccessible routes are not used, because the inaccessibility is not due to flood.
    # This is because part of the edges are disconnected from the main road network.

    r_list = []
    for k, v in flood_period.items():
        no_route_flood = [
            e[1].split('"')[1] for e in v if e[1].startswith('No route for')
        ]
        detach_flood = [
            e[1].split('"')[1] for e in v if e[1].startswith('Need at least 2 valid stops')
        ]
        r_list.append(no_route_flood + detach_flood)
    routes = [e for l in r_list for e in l]

    incident_id_list = [np.int64(i.split('-')[1]) for i in routes]

    incidents.loc[incidents['incident_id'].isin(incident_id_list), 'Total_Seconds'] = fill_value

    return incidents

