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
    data['Country'] = 'USA'
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
    data = pd.read_csv(address)

    time_cols = [
        'Call Date and Time', 'Entry Date and Time', 'Dispatch Date and Time',
        'En route Date and Time', 'On Scene Date and Time', 'Close Date and Time',
    ]
    for col in time_cols:
        data[col] = pd.to_datetime(data[col], format="%Y-%m-%d %H:%M:%S")

    data['DispatchTime'] = (data['Dispatch Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['EnRouteTime'] = (data['En route Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['TravelTime'] = (data['On Scene Date and Time'] - data['En route Date and Time']).astype("timedelta64[s]")
    data['ResponseTime'] = (data['En route Date and Time'] - data['Call Date and Time']).astype("timedelta64[s]")
    data['HourInDay'] = data['Call Date and Time'].dt.hour
    data['DayOfWeek'] = data['Call Date and Time'].dt.dayofweek

    data = data.set_index('Call Date and Time')
    data = data.sort_index()
    data = data.reset_index()

    data['IncidentPoint'] = gpd.GeoSeries(gpd.points_from_xy(
        y=data['IncidentLat'], x=data['IncidentLon'],
    ), index=data.index, crs="EPSG:4326")
    data = gpd.GeoDataFrame(data, geometry=data['IncidentPoint'])

    return data


def incidents_add_rescue_station(data, rescue):
    rescue = rescue.drop('geometry', axis=1)
    data = data[data['Rescue Squad Number'].isin(rescue.Number.to_list())]
    data = data.merge(rescue, how='left', left_on='Rescue Squad Number', right_on='Number')
    return data


def legacy__organize_data(data):
    data['CallDateTime'] = data.index
    data = data.reset_index(drop=True)
    data = data.loc[:, [
                           'Call Priority',
                           'CallDateTime', 'EntryDateTime', 'DispatchDateTime', 'EnRouteDateTime', 'OnSceneDateTime',
                           'CloseDateTime',
                           'DispatchTime', 'EnRouteTime', 'TravelTime', 'ResponseTime', 'HourInDay', 'DayOfWeek',
                           'Rescue Squad Number', 'geometry',
                           'Address', 'IncidentFullInfo', 'IncidentPoint', ]]
    data = data.rename(columns={"geometry": "RescueSquadPoint",
                                "Address": "IncidentAddress",
                                'Rescue Squad Number': 'RescueSquadNumber',
                                'Call Priority': 'CallPriority'})
    data.set_geometry("IncidentPoint")
    return data


def legacy__string_2_points(data, column, crs, index):
    x = [float(location.replace('POINT (', '').replace(')', '').split(' ')[0]) for location in
         list(data[column].values)]
    y = [float(location.replace('POINT (', '').replace(')', '').split(' ')[1]) for location in
         list(data[column].values)]
    return gpd.GeoSeries(gpd.points_from_xy(x=x, y=y), crs=crs, index=index)


def legacy_reload_incidents(address):
    data = pd.read_csv(address, index_col='CallDateTime')
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
    data = gpd.GeoDataFrame(data)
    data['RescueSquadPoint'] = legacy__string_2_points(data, 'RescueSquadPoint', "EPSG:4326", data.index)
    data['IncidentPoint'] = legacy__string_2_points(data, 'IncidentPoint', "EPSG:4326", data.index)
    return data