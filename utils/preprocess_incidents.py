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


def import_incident(address: str):
    """
    :param address: the address of geocoded and ready-to-use incident data
    :return: processed data in GeoDataFrame format
    """
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
    data['ResponseTime'] = (data['On Scene Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['HourInDay'] = data['Call Date and Time'].dt.hour
    data['DayOfWeek'] = data['Call Date and Time'].dt.dayofweek

    data = data.set_index('Call Date and Time')
    data = data.sort_index()
    data = data.reset_index()

    if 'IncidentPoint' not in data.columns:
        data['IncidentPoint'] = gpd.GeoSeries(gpd.points_from_xy(
            y=data['IncidentLat'], x=data['IncidentLon'],
        ), index=data.index, crs="EPSG:4326")
        data = gpd.GeoDataFrame(data, geometry=data['IncidentPoint'], crs="EPSG:4326")
    else:
        data = gpd.GeoDataFrame(data, geometry=data['IncidentPoint'], crs="EPSG:4326")

    minx, miny, maxx, maxy = [-76.227826, 36.550525, 75.867555, 36.931973]
    bbox = box(minx, miny, maxx, maxy)
    data = data[data.intersects(bbox)]

    data['incident_id'] = list(range(1, len(data) + 1))

    return data


def import_raw_incident_chs(address):
    df = pd.read_csv(address)
    df['id'] = range(1, len(df) + 1)

    df = df.drop(columns=['Incident_Number', 'Radio_Name'])
    df = df[~df.duplicated(subset=df.columns.difference(['id']))]

    df = df.rename(columns={
        'Response_Date': 'call_time', 'Address2': 'address',
        'Time_Assigned': 'assign_time', 'Time_Enroute': 'enroute_time',
        'Time_AtScene': 'on_scene_time', 'Time_Cleared': 'close_time',
    })

    for c in ['call_time', 'assign_time', 'enroute_time', 'on_scene_time', 'close_time']:
        df[c] = pd.to_datetime(df[c], format='%m/%d/%Y %H:%M')

    df['address'] = df['address'].str.split(' - ').str[1] + ', CHARLESTON, SC, USA'
    df = df[['id', 'address'] + [col for col in df.columns if not col in ['address', 'id']]]
    return df


def geocode_incident_chs(inc_dir, save_addr, time_range=None):
    import os
    if os.path.exists(save_addr):
        print('Geocoded file exists.')
        return None

    print('Geocoding progressing.')
    import geopy as gpy
    from geopy.extra.rate_limiter import RateLimiter

    locator = gpy.geocoders.ArcGIS()
    geocode = RateLimiter(locator.geocode, min_delay_seconds=0.1)

    data = import_raw_incident_chs(inc_dir)
    if time_range is not None:
        data = data[(data['call_time'] >= time_range[0]) & (data['call_time'] <= time_range[1])]

    data['coordinate'] = data['address'].copy().apply(geocode).apply(
        lambda loc: tuple(loc.point) if loc else None
    )
    data[['lat', 'lon', 'elevation']] = pd.DataFrame(
        data['coordinate'].copy().tolist(), index=data.index,
    )
    data.to_csv(save_addr, index=False)
    return


def import_incident_chs(address):
    df = pd.read_csv(address)
    df = df.drop(columns=['address', 'coordinate', 'elevation'])
    for c in ['call_time', 'assign_time', 'enroute_time', 'on_scene_time', 'close_time']:
        df[c] = pd.to_datetime(df[c])

    from shapely.geometry import Point
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf = gdf.set_crs(epsg=4326)

    from shapely.geometry import box
    bbox = (-79.98, 32.76, -79.92, 32.84)
    bbox = box(*bbox)
    gdf = gdf[gdf.geometry.within(bbox)]
    return gdf


def import_incidents_add_info(dir_incidents, rescue_station, period_dict, routing_nearest=None):
    incidents = import_incident(dir_incidents)
    incidents = add_actual_rescue_station(incidents, rescue_station)
    if routing_nearest is not None:
        incidents = add_nearest_rescue_station(
            incidents, rescue_station, routing_nearest=routing_nearest, mode='routing'
        )
    else:
        incidents = add_nearest_rescue_station(incidents, rescue_station, mode='geometric')
    incidents = add_period_label(incidents, 'Call Date and Time', period_dict)
    return incidents


def add_actual_rescue_station(incident, rescue):
    rescue = rescue.drop('geometry', axis=1)
    incident = incident[incident['Rescue Squad Number'].isin(rescue.Number.to_list())]
    incident = incident.merge(rescue, how='left', left_on='Rescue Squad Number', right_on='Number')
    return incident


def add_nearest_rescue_station(incident, rescue, routing_nearest=None, mode='routing'):
    if mode == 'routing':
        assert routing_nearest is not None, "Specify nearest stations with routing dist. using external file. "

    original_crs = incident.crs
    incident = incident.to_crs('epsg:32633')
    rescue = rescue.to_crs('epsg:32633')
    assert rescue.crs == incident.crs

    if mode == 'geometric':
        incident_new = incident.sjoin_nearest(rescue, how='left', lsuffix='actual', rsuffix='nearest')
        incident_new = incident_new.to_crs(original_crs)
    elif mode == 'routing':
        import pandas as pd
        nearest_station = pd.read_csv(routing_nearest)
        time_cols = ['Call_Date_and_Time', 'Entry_Date_and_Time', 'Dispatch_Date_and_Time']
        for col in time_cols:
            nearest_station[col] = pd.to_datetime(nearest_station[col])
            nearest_station[col] = nearest_station[col].dt.tz_localize('America/New_York')
            nearest_station[col.replace('_', ' ')] = nearest_station[col].dt.tz_convert('UTC')
        incident_new = incident.merge(
            nearest_station[[i.replace('_', ' ') for i in time_cols] + ['Number']],
            how='left', suffixes=('_actual', '_nearest'),
            on=['Call Date and Time', 'Entry Date and Time', 'Dispatch Date and Time'],
        )
        assert len(incident) == len(incident_new), "DFs are inconsistent after merge."
    else:
        raise ValueError(f'Mode not specified')
    return incident_new


def add_period_label(incident, time_col, label_dict):
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
            (incident[time_col] >= t_1) & (incident[time_col] <= t_2),
            'period_label'] = l
        incident.loc[
            (incident[time_col] >= t_1) & (incident[time_col] <= t_2),
            'period_actual'] = pd.Series([pd.Timestamp(t_1), pd.Timestamp(t_2)]).mean()

    return incident


def convert_feature_class_to_df(incident, feature_class_addr, label_list, mode_label=''):
    """
    The function reads the results in the ArcGIS model and converts the results into DataFrame format
    :param incident: incident information w/o travel time estimation
    :param feature_class_addr: the address of the ArcGIS model
    :param label_list: the labels of the layers in the ArcGIS model
    :param mode_label: optional, additional information
    :return: incident information with travel time estimation
    """
    import arcpy
    import warnings

    df_list = []
    for l in label_list:
        if arcpy.Exists(f'{feature_class_addr}_{l}') is not True:
            # warnings.warn(f'route_results_{l}{mode_label} is missing')
            continue

        arr = arcpy.da.FeatureClassToNumPyArray(
            f'{feature_class_addr}_{l}{mode_label}', ('Name', 'Total_Seconds', 'Total_Miles')
        )
        df = pd.DataFrame(arr)
        df['rescue_name'] = df['Name'].str.split('-').str[0]
        df['incident_id'] = df['Name'].str.split('-').str[1].astype('int64')
        df_list.append(df)

    df = pd.concat(df_list, axis=0)
    incident = incident.merge(df[['incident_id', 'Total_Seconds', 'Total_Miles']], how='left', on='incident_id')

    return incident


def add_inaccessible_routes(incidents, addr_inaccessible_routes: str, fill_value=-999):
    """
    In ArcGIS analysis, inaccessible routes are not saved in the results. So, the error reports of
    ArcGIS were saved and read here to add the information to the incident DataFrame.
    Note that normal-time inaccessible routes are not used, because the inaccessibility is not due to flood.
    This is because part of the edges are disconnected from the main road network.
    :param incidents: DataFrame of the incident information
    :param addr_inaccessible_routes: error reports address
    :param fill_value: NULL value
    :return: DataFrame of the incident information with added information
    """
    import json
    import numpy as np

    with open(addr_inaccessible_routes, 'r') as f:
        inaccessible_routes = json.load(f)
    flood_period = inaccessible_routes['flood']

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


def convert_timedelta_2_seconds(incidents, col_names):
    for col in col_names:
        incidents[col] = incidents[col].dt.total_seconds()
    return incidents


def add_geo_unit(incidents, dir_unit, id_col_geo):
    """
    Add geographic information to the incident DataFrame.
    :param incidents: DataFrame, incident information
    :param dir_unit: address of the geographic information
    :param id_col_geo: the name of geographic information column
    :return: DataFrame, incident information
    """
    unit_geo = gpd.read_file(dir_unit)
    unit_geo = unit_geo[id_col_geo + ['geometry']]
    unit_geo = unit_geo.to_crs(incidents.crs)
    incidents = incidents.sjoin(unit_geo, how='left')
    return incidents


def import_demographic(dir_demo, value_col, remove_tract=None):
    demographic = pd.read_csv(dir_demo)[['GEO_ID', 'NAME'] + value_col].iloc[1:]

    if 'Block Group' in demographic.iloc[0]['NAME']:
        demographic['block_group_name'] = demographic['NAME'].str.extract(r'Block Group (\d+\.?\d*)')
    demographic['tract_name'] = demographic['NAME'].str.extract(r'Census Tract (\d+\.?\d*)')
    demographic['city_name'] = demographic['NAME'].str.extract(r'Census Tract \d+\.?\d*, ([^,]+)')
    assert (demographic['city_name'] == 'Virginia Beach city').all()

    if remove_tract is not None:
        demographic = demographic[~demographic['tract_name'].isin(remove_tract)]

    demographic = demographic[demographic[value_col[0]] != '-']

    demographic['demographic_value'] = demographic[value_col].astype(float)
    if len(value_col) > 1:
        demographic['demographic_value'] = demographic[value_col].sum(axis=1)

    return demographic


def merge_incidents_demographic_bg(incidents, demographic):
    demographic['tract_name_adapt'] = '0' + (demographic['tract_name'].astype(float) * 100).astype(int).astype(str)
    demographic['id'] = demographic['tract_name_adapt'] + demographic['block_group_name']
    incidents['id'] = incidents['TRACTCE'] + incidents['BLKGRPCE']
    incidents = incidents.merge(demographic, how='left', on='id')
    return incidents


def get_geo_bg(dir_bg, target_county_num='810'):
    geo = gpd.read_file(dir_bg)
    geo = geo[geo['COUNTYFP'] == target_county_num]
    return geo


def merge_demographic_geo_bg(demographic, dir_bg):
    geo = get_geo_bg(dir_bg)
    geo['id'] = geo['TRACTCE'] + geo['BLKGRPCE']

    demographic['tract_name_adapt'] = '0' + (demographic['tract_name'].astype(float) * 100).astype(int).astype(str)
    demographic['id'] = demographic['tract_name_adapt'] + demographic['block_group_name']

    demographic = demographic.merge(geo, how='left', on='id')
    demographic = gpd.GeoDataFrame(demographic, geometry='geometry')
    return demographic


def merge_geo_unit_geo_bg(geo_unit, dir_bg, target_county_num='810'):
    geo = gpd.read_file(dir_bg)
    geo = geo[geo['COUNTYFP'] == target_county_num]
    geo['id'] = geo['TRACTCE'] + geo['BLKGRPCE']

    geo_unit = geo_unit.merge(geo, how='left', on='id')
    geo_unit = gpd.GeoDataFrame(geo_unit, geometry='geometry')
    return geo_unit


def aggr_incidents_geo(incidents, period_dict, dir_bg_boundaries):
    g_units = incidents[
        ['id', 'diff_travel', 'demographic_value', 'wellness', 'period_actual', 'TravelTime', 'Total_Seconds']
    ].groupby('id').mean()

    g_units = add_period_label(g_units, 'period_actual', period_dict)
    g_units['period_actual'] = g_units['period_actual'].apply(
        lambda x: f"{f'{x.hour:02}:00'} - {f'{(x.hour + 1) % 24:02}:00'}"
    )

    g_units = merge_geo_unit_geo_bg(g_units, dir_bg_boundaries)
    return g_units


def delete_outlier_zscore(df, col, threshold=3):
    from scipy import stats
    import numpy as np

    for c in col:
        df = df[
            (np.abs(stats.zscore(df[c].values)) < 3)
        ]

    return df

def delete_outlier_mahalanobis(df, col):
    import numpy as np
    df_c = df[col]

    mean = np.mean(df_c.values, axis=0).reshape(1, -1)
    inv_cov_matrix = np.linalg.inv(np.cov(df_c, rowvar=False))

    def mahalanobis_distance(x, mean, inv_cov_matrix):
        x_minus_mean = x - mean
        left_term = np.dot(x_minus_mean, inv_cov_matrix)
        mahal = np.dot(left_term, x_minus_mean.T)
        return mahal.diagonal()

    df_c.loc[:, ['Mahalanobis']] = df_c.apply(
        lambda row: mahalanobis_distance(row.values.reshape(1, -1), mean, inv_cov_matrix),
        axis=1
    )
    threshold = 5.991  # Corresponds to chi-squared value with 2 degrees of freedom at p=0.05
    df = df[df_c['Mahalanobis'] <= threshold]
    return df


def add_econ_class(gdf, bar: float, income_col, save_col, dir_save=None):
    """
    Add economic class (lower-income class; middle-higher income class) to incident data
    :param gdf: incidents
    :param bar: threshold of the middle class
    :param income_col: name of the column showing income
    :param save_col: column to save
    :param dir_save: address to save
    :return: incidents with added information
    """
    gdf['econ_class'] = gdf[income_col].apply(lambda x: 'middle_higher' if x > bar else 'lower')
    if dir_save is not None:
        gdf[save_col + ['econ_class', 'geometry']].to_file(dir_save, driver="ESRI Shapefile")
    return gdf


def save_ict_to_shp(gdf_i, dir_save, time_range_dict=None, to_arcgis=False, gdb_path=None,):

    if time_range_dict is not None:
        begin, end = list(time_range_dict.keys())
        gdf = gdf[(gdf['Call Date and Time'] >= begin) & (gdf['Call Date and Time'] <= end)]

    for col in gdf.select_dtypes(['datetime64[ns, UTC]']).columns:
        gdf[col] = gdf[col].astype(str)
    for col in gdf.select_dtypes(['datetime64[ns]']).columns:
        gdf[col] = gdf[col].astype(str)

    for col in gdf.select_dtypes(['timedelta64[ns]']).columns:
        gdf[col] = gdf[col].dt.total_seconds()

    geometry_cols = [col for col in gdf.columns if isinstance(gdf[col], gpd.GeoSeries)]
    geometry_cols_remove = [i for i in geometry_cols if i != 'geometry']
    gdf = gdf.drop(columns=geometry_cols_remove)

    gdf.to_file(f'{dir_save}/flood_ict_with_travel_time_estimate.shp')
    gdf.to_file(f'{dir_save}/flood_ict_with_travel_time_estimate.geojson')

    if to_arcgis is True:
        assert gdb_path is not None
        import arcpy
        arcpy.env.workspace = gdb_path
        arcpy.FeatureClassToGeodatabase_conversion(
            f'{dir_save}/flood_ict_with_travel_time_estimate.shp', gdb_path
        )

    return


def create_grid(gdf, n_rows=10, n_cols=10):
    from shapely.geometry import box
    xmin, ymin, xmax, ymax = gdf.total_bounds
    cell_width = (xmax - xmin) / n_cols
    cell_height = (ymax - ymin) / n_rows

    grid_cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            x1 = xmin + i * cell_width
            y1 = ymin + j * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            grid_cells.append({
                "geometry": box(x1, y1, x2, y2),
                "label": f"{i + 1}-{j + 1}"  # Assign label as "col-row"
            })
    grid = gpd.GeoDataFrame(grid_cells, crs=gdf.crs)
    return grid


def get_hotspots_ave_time(icd):
    icd['Call Date and Time'] = icd['Call Date and Time'].dt.tz_convert('America/New_York')
    icd['local_hour'] = icd['Call Date and Time'].dt.hour
    hotspots = icd['IncidentPoint'].value_counts()

    t_by_h = []
    t_min_by_h = []
    t_std_by_h = []
    loc = []
    for h in hotspots[hotspots >= 24].index:
        origins = icd[icd['IncidentPoint'] == h]['Rescue Squad Number'].value_counts()
        incidents_select = icd[
            (icd['IncidentPoint'] == h) & (icd['Rescue Squad Number'] == origins.idxmax())
            ]
        time_h = incidents_select.groupby('local_hour')['TravelTime'].mean()
        if len(time_h) == 24:
            loc.append(h)
            t_by_h.append(time_h.to_list())
            t_min_by_h.append(
                incidents_select.groupby('local_hour')['Total_Seconds'].mean().to_list()
            )
            t_std_by_h.append(
                incidents_select.groupby('local_hour')['TravelTime'].std().to_list()
            )
    time_by_hour = [sum(values) / len(values) for values in zip(*t_by_h)]
    time_min_by_hour = [sum(values) / len(values) for values in zip(*t_min_by_h)]
    assert all(element == time_min_by_hour[0] for element in time_min_by_hour)
    import numpy as np
    time_std_by_hour = np.array(t_std_by_h)
    time_std_by_hour = np.nan_to_num(time_std_by_hour, nan=0) ** 2
    time_std_by_hour = np.sum(time_std_by_hour, axis=0)
    time_std_by_hour = np.sqrt(time_std_by_hour)
    return time_by_hour, time_std_by_hour, time_min_by_hour, loc


