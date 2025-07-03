import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import rasterio


def pull_roads_osm(city, city_abbr, folder, filter):
    """
    Download and save OSM road network data for a city using OSMnx.

    Args:
        city (str or tuple): City name or bounding box for OSMnx.
        city_abbr (str): Abbreviation for the city (used in filenames).
        folder (str): Directory to save the output files.
        filter (str): OSM filter string for road types.
    """
    import osmnx as ox
    import networkx as nx

    if isinstance(city, str):
        graph = ox.graph_from_place(
            city, custom_filter=f'["highway"~{filter}]',
            simplify=False, retain_all=True, truncate_by_edge=True
        )
    elif isinstance(city, tuple):
        graph = ox.graph_from_bbox(
            bbox=city, custom_filter=f'["highway"~{filter}]',
            simplify=False, retain_all=True, truncate_by_edge=True
        )
    else:
        raise ValueError('city_abbr must be str or tuple')
    graph = ox.simplify_graph(graph, edge_attrs_differ=['osmid', 'oneway'])

    gdf_nodes = ox.graph_to_gdfs(graph, edges=True)[0]
    gdf_edges = ox.graph_to_gdfs(graph, edges=True)[1]

    for u, v, data in graph.edges(data=True):
        if 'geometry' in data.keys():
            del data['geometry']
    nx.write_gml(graph, f'{folder}/graph_{city_abbr}.gml')

    gdf_nodes.to_file(f"{folder}/road_intersection_{city_abbr}.geojson", driver="GeoJSON")
    gdf_edges.to_file(f"{folder}/road_segment_{city_abbr}.geojson", driver="GeoJSON")


def import_roads_osm(folder, city_abbr):
    """
    Import OSM road network data (nodes, edges, and graph) from files.

    Args:
        folder (str): Directory containing the OSM files.
        city_abbr (str): Abbreviation for the city (used in filenames).

    Returns:
        tuple: (graph, gdf_edges, gdf_nodes)
    """
    import networkx as nx

    gdf_nodes = gpd.read_file(f"{folder}/road_intersection_{city_abbr}.geojson")
    gdf_edges = gpd.read_file(f"{folder}/road_segment_{city_abbr}.geojson")
    graph = nx.read_gml(f'{folder}/graph_{city_abbr}.gml')
    return graph, gdf_edges, gdf_nodes


def add_roads_max_speed(gdf, speed, maxspeed_name):
    """
    Assign maximum speed to each road segment based on its highway type.

    Args:
        gdf (GeoDataFrame): Road segments.
        speed (dict): Mapping from highway type to speed.
        maxspeed_name (str): Name of the new column for max speed.

    Returns:
        GeoDataFrame: Road segments with max speed assigned.
    """
    gdf[maxspeed_name] = gdf['highway'].map(speed)
    assert gdf['highway'].isna().any() == False, 'Speed was not assigned to all roads.'
    return gdf


def import_turn_restriction(addr):
    """
    Import and process turn restriction data from a GeoJSON file.

    Args:
        addr (str): Path to the turn restriction GeoJSON file.

    Returns:
        DataFrame: Processed turn restriction data.
    """
    import ast

    gdf = gpd.read_file(addr)
    gdf_copy = gdf.copy()
    gdf = gdf[['@id', '@relations',]]
    gdf['@id'] = gdf['@id'].str.split('/').str[1]

    gdf['@relations'] = gdf['@relations'].apply(ast.literal_eval)
    gdf = gdf.explode('@relations')
    gdf['@rel'] = gdf['@relations'].apply(lambda x: x['rel'])
    gdf['@role'] = gdf['@relations'].apply(lambda x: x['role'])
    gdf = gdf[['@id', '@rel', '@role']]

    row_list = []
    for rel, group in gdf.groupby('@rel'):
        assert group['@role'].isin(['from', 'via', 'to']).all(), 'Triple incomplete.'
        assert len(group[group['@role'] == 'from']) == 1, "Multiple 'from'."
        assert len(group[group['@role'] == 'to']) == 1, "Multiple 'to'."
        """
            Multiple 'from' or 'to' are only possible when there is 'no entry' or 'no exit' restriction.
        """

        row = [
            rel,
            group[group['@role'] == 'from']['@id'].iloc[0],
            group[group['@role'] == 'via']['@id'].to_list(),
            group[group['@role'] == 'to']['@id'].iloc[0],
        ]
        row_list.append(row)
    gdf_out = pd.DataFrame(row_list, columns=['rel', 'from', 'via', 'to'])

    #### Enhancement for ArcGIS Pro ####
    #### output w/o the following codes would still be valid ####
    from shapely.ops import unary_union

    gdf_copy['object_id'] = gdf_copy['@id'].str.split('/').str[1]
    gdf_copy['object_type'] = gdf_copy['@id'].str.split('/').str[0]
    type_checker = gdf_copy.set_index('object_id')['object_type'].to_dict()
    gdf_out['via_node'] = np.nan
    gdf_out['via_node'] = pd.Series(dtype='object')
    for i, row in gdf_out.iterrows():

        assert type_checker[row['from']] == 'way', "'from' must be a 'way'"
        assert type_checker[row['to']] == 'way', "'to' must be a 'way'"

        v_way_list = []
        v_node_list = []
        for v in row['via']:
            if type_checker[v] == 'way':
                v_way_list.append(v)
            elif type_checker[v] == 'node':
                if v == '2050904603':
                    print()
                v_node_list.append(v)
        gdf_out.at[i, 'via'] = v_way_list
        assert len(v_node_list) <= 1, 'Restriction can via only one node or no node.'
        gdf_out.at[i, 'via_node'] = v_node_list

        geo_from = gdf_copy[gdf_copy['object_id'] == row['from']]['geometry'].iloc[0]
        geo_via_list = []
        for v in v_way_list:
            geo_via_list.append(gdf_copy[gdf_copy['object_id'] == v]['geometry'].iloc[0])
        geo_to = gdf_copy[gdf_copy['object_id'] == row['to']]['geometry'].iloc[0]
        combined_geo = unary_union([geo_from] + [e for e in geo_via_list] + [geo_to])
        gdf_out.loc[i, 'geometry'] = combined_geo

    return gdf_out


def import_bridge_tunnel(addr):
    """
    Import bridge and tunnel data from a GeoJSON file.

    Args:
        addr (str): Path to the bridge/tunnel GeoJSON file.

    Returns:
        DataFrame: Processed bridge/tunnel data.
    """
    gdf = gpd.read_file(addr)
    gdf['id'] = gdf['id'].str.split('/').str[1]
    gdf = gdf[['id', 'highway']]
    return gdf


def add_travel_time_2_seg(roads, maxspeed_name, travel_time_name):
    """
    Calculate travel time for each road segment based on its length and max speed.

    Args:
        roads (GeoDataFrame): Road segments.
        maxspeed_name (str): Column name for max speed.
        travel_time_name (str): Name of the new column for travel time.

    Returns:
        GeoDataFrame: Road segments with travel time assigned.
    """
    # input('For add_travel_time_2_seg, input speed is mile/h, and output time is seconds.')
    roads['maxspeed_assigned_m_per_s'] = roads[maxspeed_name] * 1.60934 * 1000 / 60 / 60
    roads[travel_time_name] = roads['length'] / roads['maxspeed_assigned_m_per_s']
    return roads


def add_water_depth_on_roads_w_bridge(
        roads, inundation_tif_addr, label, new_col_name,
        remove_bridge=True,
        remove_inundation_under_bridge=False, geo_w_bridge=None,
):
    """
    Add water depth information to road segments, considering bridges and inundation.

    Args:
        roads (GeoDataFrame): Road segments.
        inundation_tif_addr (str): Path to the inundation raster file.
        label (int or str): Label for the inundation period.
        new_col_name (dict): Mapping for new columns (e.g., {'max': 'max_depth', 'mean': 'mean_depth'}).
        remove_bridge (bool): Whether to remove bridge segments from inundation.
        remove_inundation_under_bridge (bool): Remove inundation under bridges.
        geo_w_bridge (GeoDataFrame, optional): Road segments with bridge info.

    Returns:
        GeoDataFrame: Road segments with water depth columns added.
    """
    from rasterio.features import shapes
    import warnings

    assert not (remove_bridge and remove_inundation_under_bridge)
    if remove_inundation_under_bridge:
        assert geo_w_bridge is not None

    inundation = rasterio.open(inundation_tif_addr)
    results = (
        {'properties': {'val': v}, 'geometry': s} for i, (s, v)
        in enumerate(shapes(inundation.read(1), mask=None, transform=inundation.transform))
    )

    polygons = gpd.GeoDataFrame.from_features(list(results))
    polygons = polygons.set_crs(inundation.crs)
    polygons = polygons.to_crs(roads.crs)
    assert polygons.crs == roads.crs, 'CRS inconsistent.'

    polygons = polygons[polygons['val'] != polygons['val'].max()]
    warnings.warn('It is assumed that the "no" data value was set to a very big number by ArcGIS.')

    if remove_inundation_under_bridge:  # delete inundations under bridges
        geo_w_bridge = geo_w_bridge.to_crs(roads.crs)
        bridge_geo = geo_w_bridge[~geo_w_bridge['bridge'].isna()]
        non_bridge_geo = geo_w_bridge[geo_w_bridge['bridge'].isna()]

        # overhead bridge location is untouched, road under overhead bridge could be inundated
        # the inundation calculation for the overhead bridge is not correct
        bridge_covered = bridge_geo.loc[
            bridge_geo.sjoin(polygons, how='inner', predicate='intersects').index.unique()
        ]
        overhead_bridge = get_overhead_bridge(bridge_covered, non_bridge_geo, plot=False)

        polygons_c = polygons.copy()
        inundation_2_go = bridge_geo[~bridge_geo.index.isin(overhead_bridge)].sjoin(
            polygons.reset_index(), how='inner', predicate='intersects'
        )
        polygons = polygons[~polygons.index.isin(inundation_2_go['index_right'].tolist())]

        # calculate only for overhead bridge
        polygons_c = polygons_c[~polygons_c.index.isin(
            bridge_geo.sjoin(
                polygons_c.reset_index(), how='inner', predicate='intersects'
            )['index_right'].tolist()
        )]

        ov_bridge_s2_list = []
        for o in overhead_bridge:
            ov_bridge = bridge_covered[bridge_covered.index == o]
            assert len(ov_bridge) == 1
            ov_bridge_s2_info = get_corresponding_road(ov_bridge, roads, top=1)
            ov_bridge_s2_list.append([i[0] for i in ov_bridge_s2_info])

        ov_bridge_s2 = roads[roads.index.isin([i for l in ov_bridge_s2_list for i in l])]
        ov_bridge_s2 = get_water_depth(ov_bridge_s2, polygons_c, new_col_name, label)

    roads = get_water_depth(roads, polygons, new_col_name, label)

    if remove_bridge is True:
        roads.loc[roads['bridge'] == 'yes', f'{new_col_name["max"]}_{label}'] = np.nan
        roads.loc[roads['bridge'] == 'yes', f'{new_col_name["mean"]}_{label}'] = np.nan

    if remove_inundation_under_bridge:
        roads.iloc[ov_bridge_s2.index] = ov_bridge_s2.iloc[0]

    return roads


def get_water_depth(roads, polygons, new_col_name, label):
    """
    Calculate max and mean water depth for each road segment based on inundation polygons.

    Args:
        roads (GeoDataFrame): Road segments.
        polygons (GeoDataFrame): Inundation polygons.
        new_col_name (dict): Mapping for new columns.
        label (int or str): Label for the inundation period.

    Returns:
        GeoDataFrame: Road segments with water depth columns added.
    """
    inundated_roads = roads.copy().sjoin(polygons, how='inner', predicate='intersects')

    inundated_roads_max = inundated_roads[['val']].groupby(inundated_roads.index).max()
    inundated_roads_mean = inundated_roads[['val']].groupby(inundated_roads.index).mean()

    roads_c = roads.copy()
    roads_c.loc[:, f"{new_col_name['max']}_{label}"] = inundated_roads_max['val']
    roads_c.loc[:, f"{new_col_name['mean']}_{label}"] = inundated_roads_mean['val']

    return roads_c


def get_overhead_bridge(bridge_geo, non_bridge_geo, plot=True):
    """
    Identify overhead bridges by checking which bridges cross non-bridge roads.

    Args:
        bridge_geo (GeoDataFrame): Bridge road segments.
        non_bridge_geo (GeoDataFrame): Non-bridge road segments.
        plot (bool): Whether to plot the results for visual inspection.

    Returns:
        list: Indices of overhead bridges.
    """
    if plot:
        import matplotlib.pyplot as plt
    overhead_bridge_list = []
    for i, row in bridge_geo.iterrows():
        crossed = non_bridge_geo[non_bridge_geo.crosses(row['geometry'])]
        if len(crossed) != 0:
            if plot:
                gdf_row = gpd.GeoDataFrame(pd.DataFrame([row]), crs=bridge_geo.crs)
                crossed_roads = pd.concat([gdf_row, crossed], ignore_index=True)
                crossed_roads.plot()
                plt.show()
            warnings.warn(f'Overhead bridge with index {i} is overlapped with inundation.')
            overhead_bridge_list.append(i)
    return overhead_bridge_list


def get_corresponding_road(road, roads, top=1, bf=10):
    """
    Find the corresponding roads for a given road segment.

    Args:
        road (GeoDataFrame): Single road segment.
        roads (GeoDataFrame): All road segments.
        top (int): Number of top matches to return.
        bf (float): Buffer size for spatial join.

    Returns:
        DataFrame: Candidate matching road segments.
    """
    assert isinstance(road, gpd.GeoDataFrame)
    assert isinstance(roads, gpd.GeoDataFrame)
    assert len(road) == 1

    road = road.to_crs('epsg:2284')
    road.loc[:, 'geometry'] = road.geometry.buffer(bf)

    roads = roads.to_crs(road.crs)
    candidate_roads = gpd.sjoin(roads, road, how='inner', predicate='intersects')

    overlapping_degree = []
    for i, c in candidate_roads.iterrows():
        overlap = road.iloc[0:1].geometry.intersection(c.geometry).length.values[0]
        overlapping_degree.append([i, overlap])

    overlapping_degree_sorted = sorted(overlapping_degree, key=lambda x: x[1], reverse=True)
    return overlapping_degree_sorted[:top]


class InundationToSpeed:
    """
    Class to convert water depth to speed reduction for road segments.
    
    This class implements methods to calculate how water depth affects vehicle speed
    on road segments, including cutoff thresholds and speed reduction curves.
    """
    
    def __init__(self, thr=30, unit_checker=False):
        """
        Initialize the InundationToSpeed converter.

        Args:
            thr (float, optional): Water depth threshold in cm. Defaults to 30.
            unit_checker (bool, optional): Whether to prompt for unit confirmation. Defaults to False.
        """
        # input('threshold should be cm.')
        self.thr = thr
        self.unit_checker = unit_checker

        self.speed_unique = None
        self.curves = {}

        self.cut_series = None
        self.reduce_series = None
        return

    def build_decreasing_curve(self):
        """
        Build linear regression curves for speed reduction based on water depth.
        
        Creates a linear model for each unique speed value to predict reduced speed
        based on water depth from 0 to threshold.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        for s in self.speed_unique:
            x = np.array([0, self.thr])[:, np.newaxis]
            y = np.array([s, 0])[:, np.newaxis]
            model = LinearRegression()
            model.fit(x, y)
            self.curves[s] = model
        return

    def calculate_safe_control_speed(self, series):
        """
        Calculate safe control speed based on water depth using a quadratic formula.

        Args:
            series (Series): Water depth values.

        Returns:
            Series: Safe control speed values.
        """
        return 0.0009 * series * series - 0.5529 * series + 86.9448

    def cutoff(self, depth_series):
        """
        Determine which road segments should be cut off due to excessive water depth.

        Args:
            depth_series (Series): Water depth values in feet.

        Returns:
            Series: Boolean series indicating which segments should be cut off.
        """
        if self.unit_checker:
            input('For cutoff(), the input water depth should be ft. Enter to continue.')

        if_cut = depth_series * 30.48 >= self.thr
        self.cut_series = if_cut
        return if_cut

    def reduce(self, depth_series, maxspeed_series):
        """
        Calculate reduced speed for road segments based on water depth.

        Args:
            depth_series (Series): Water depth values in feet.
            maxspeed_series (Series): Original maximum speed values in mph.

        Returns:
            Series: Reduced speed values.
        """
        if self.unit_checker:
            input('For reduce(), the input water depth should be ft, speed is mile/h. Enter to continue.')

        speed_df = pd.DataFrame(index=maxspeed_series.index)
        speed_df['speed'] = [np.nan] * len(speed_df)

        maxspeed_series = maxspeed_series[(depth_series * 30.48 < self.thr) & (depth_series * 30.48 > 0)]
        depth_series = depth_series[(depth_series * 30.48 < self.thr) & (depth_series * 30.48 > 0)]
        speed_df['depth'] = depth_series

        self.speed_unique = maxspeed_series.unique()
        self.build_decreasing_curve()

        for s in self.speed_unique:
            curve = self.curves[s]
            s_depth_series = depth_series[maxspeed_series == s]
            s_speed = curve.predict(
                (s_depth_series * 30.48).values[:, np.newaxis]
            )
            speed_df.loc[s_depth_series.index, 'speed'] = s_speed[:, 0]

        # safe control theoretical limit
        speed_df['safe_control'] = self.calculate_safe_control_speed(
            speed_df['depth'].fillna(0) * 30.48 * 10
        ) * 0.621371
        speed_df['speed'] = speed_df['speed'].where(
            (speed_df['speed'] <= speed_df['safe_control']) | speed_df['speed'].isna(),
            speed_df['safe_control']
        )

        self.reduce_series = speed_df['speed']
        return speed_df['speed']

    def apply_orig_speed(self, original_maxspeed_series):
        """
        Apply original speed to road segments that are not affected by inundation.

        Args:
            original_maxspeed_series (Series): Original maximum speed values in mph.

        Returns:
            Series: Final speed values after applying cutoff and reduction.
        """
        if self.unit_checker:
            input('For apply_original_speed(), the input speed is mile/h. Enter to continue.')

        speed_series = self.cut_series.map({True: 1e-5, False: np.nan})
        speed_series = speed_series.fillna(self.reduce_series)
        speed_series = speed_series.fillna(original_maxspeed_series)

        return speed_series


def import_road_seg_w_inundation_info(
        dir_road_inundated, speed_assigned, VDOT_speed=None, VDOT_speed_col=None, osm_match_vdot=None
):
    """
    Import road segments with inundation information and calculate travel times.

    Args:
        dir_road_inundated (str): Path to road segments with inundation data.
        speed_assigned (dict): Speed assignment dictionary.
        VDOT_speed (dict, optional): VDOT speed data. Defaults to None.
        VDOT_speed_col (str, optional): VDOT speed column name. Defaults to None.
        osm_match_vdot (str, optional): Path to OSM-VDOT matching file. Defaults to None.

    Returns:
        GeoDataFrame: Road segments with inundation-adjusted travel times.
    """
    road_segment_inundation = gpd.read_file(dir_road_inundated)
    road_segment_inundation = add_roads_max_speed(
        road_segment_inundation, speed_assigned,
        'maxspeed_assigned_mile'
    )
    road_segment_inundation = add_travel_time_2_seg(
        road_segment_inundation,
        'maxspeed_assigned_mile', 'travel_time_s'
    )

    if VDOT_speed is not None:  # if traffic simulation is involved, updated the normal speeds
        assert osm_match_vdot is not None, 'Match info is missing.'
        match_info = gpd.read_file(osm_match_vdot)

        for l in range(25, 48 + 1):
            road_segment_inundation = add_VDOT_speed(
                road_segment_inundation, VDOT_speed, l,
                match_info, VDOT_speed_col, 25,  # label 25 is at UTC 00:00
            )

    ll = ''
    converter = InundationToSpeed(30)
    for l in range(25, 48 + 1):

        if VDOT_speed is not None:  # if traffic simulation is involved, use the updated speed
            ll = f'_{l}'

        converter.cutoff(
            road_segment_inundation[f'max_depth_{l}']
        )
        converter.reduce(
            road_segment_inundation[f'mean_depth_{l}'],
            road_segment_inundation[f'maxspeed_assigned_mile{ll}'],
        )
        road_segment_inundation[f'maxspeed_inundated_mile_{l}'] = converter.apply_orig_speed(
            road_segment_inundation[f'maxspeed_assigned_mile{ll}'],
        )
        road_segment_inundation = add_travel_time_2_seg(
            road_segment_inundation,
            f'maxspeed_inundated_mile_{l}', f'travel_time_s_{l}'
        )
    return road_segment_inundation


def add_VDOT_speed(
        road_segment, VDOT_speed, time_label, match_info, VDOT_speed_col, time_label_offset=0,
):
    """
    Add VDOT speed data to road segments for a specific time period.

    Args:
        road_segment (GeoDataFrame): Road segment data.
        VDOT_speed (dict): VDOT speed data by time period.
        time_label (int): Time label for the period.
        match_info (GeoDataFrame): OSM-VDOT matching information.
        VDOT_speed_col (str): VDOT speed column name.
        time_label_offset (int, optional): Offset for time label. Defaults to 0.

    Returns:
        GeoDataFrame: Road segments with VDOT speed data added.
    """
    time_label_updated = time_label - time_label_offset
    do = False
    for k, v in VDOT_speed.items():
        start_time = int(k.split('-')[0])  # left end is closed
        end_time = int(k.split('-')[1])  # right end is open
        if end_time > start_time:  # same day
            if (time_label_updated >= start_time) and (time_label_updated < end_time):
                do = True
        if end_time < start_time:  # cross day
            if ((time_label_updated >= start_time) and (time_label_updated < 24)) or (
                    (time_label_updated >= 0) and (time_label_updated < end_time)
            ):
                do = True

        if do is True:
            v = calculate_speed_VDOT_output(v, VDOT_speed_col)
            road_segment.loc[:, f'maxspeed_assigned_mile_{time_label}'] = road_segment[
                'maxspeed_assigned_mile'
            ]
            road_segment_add_match = road_segment.merge(
                match_info[['geometry', 'vdot_id']], how='left', on='geometry'
            ).merge(
                v[['speed', 'ID']], how='left', left_on='vdot_id', right_on='ID'
            )
            road_segment.loc[
                ~road_segment_add_match['speed'].isna(), f'maxspeed_assigned_mile_{time_label}'
            ] = road_segment_add_match.loc[
                ~road_segment_add_match['speed'].isna(), 'speed'
            ]
            road_segment[f'maxspeed_assigned_mile_{time_label}'] = road_segment[
                f'maxspeed_assigned_mile_{time_label}'
            ].round(2)
    assert do is True, 'time_label is not matched with VDOT_speed info.'
    return road_segment


def calculate_speed_VDOT_output(gdf, speed_col):
    """
    Calculate speed from VDOT output data.

    Args:
        gdf (DataFrame): VDOT output data.
        speed_col (str): Speed column name ('CSPD_1' or 'TIME_1').

    Returns:
        DataFrame: Data with calculated speed values.
    """
    if speed_col == 'CSPD_1':
        gdf['speed'] = gdf[speed_col]
        return gdf
    elif speed_col == 'TIME_1':
        speed_column = gdf['DISTANCE_x'] / gdf[speed_col] * 60
        speed_column[speed_column <= 1e-1] = 0
        speed_column[speed_column >= 65] = speed_column[(speed_column < 65) & (speed_column > 1e-1)].median()
        gdf['speed'] = speed_column
        return gdf
    else:
        raise ValueError(f'speed_col {speed_col} is not supported.')


def merge_road_info_VDOT(dir_shape, dir_info):
    """
    Merge road shapefile with VDOT information from DBF file.

    Args:
        dir_shape (str): Path to road shapefile.
        dir_info (str): Path to VDOT DBF file.

    Returns:
        GeoDataFrame: Road data merged with VDOT information.
    """
    from simpledbf import Dbf5
    road_shp = gpd.read_file(dir_shape)
    road_info = Dbf5(dir_info).to_dataframe()
    road_info = road_info[road_info['COUNTY'] == 5]  # keep records for VB
    road_shp = road_shp.merge(road_info, on=['A', 'B'], how='inner', suffixes=('_shp', '_full_net'))
    road_shp = gpd.GeoDataFrame(road_shp, geometry=road_shp['geometry'])
    return road_shp


def match_osm_n_VDOT(dir_osm, dir_VDOT, dir_save, not_service=True, dir_VDOT_info=None):
    """
    Match OSM road data with VDOT road data.

    Args:
        dir_osm (str): Path to OSM road data.
        dir_VDOT (str): Path to VDOT road data.
        dir_save (str): Path to save matched data.
        not_service (bool, optional): Whether to exclude service roads. Defaults to True.
        dir_VDOT_info (str, optional): Path to VDOT info file. Defaults to None.
    """
    df_osm = gpd.read_file(dir_osm)
    df_vdot = gpd.read_file(dir_VDOT)

    if dir_VDOT_info is not None:
        df_vdot = merge_road_info_VDOT(dir_VDOT, dir_VDOT_info)
    if not_service:
        df_osm = df_osm[df_osm['highway'] != 'service']

    def get_vdot_match(row):
        match_list = get_corresponding_road(
            gpd.GeoDataFrame(row.to_frame().T, geometry='geometry').set_crs(df_osm.crs),
            df_vdot, top=1
        )
        if match_list != []:
            return match_list[0][0]
        else:
            return None

    df_osm['match_index'] = df_osm.apply(get_vdot_match, axis=1)
    df_osm = df_osm.merge(df_vdot[['ID']].reset_index(), left_on='match_index', right_on='index', how='left')
    df_osm = df_osm[['u', 'v', 'osmid', 'name', 'length', 'geometry', 'ID']]
    df_osm = df_osm.rename(columns={'ID': 'vdot_id'})
    df_osm.to_file(dir_save, driver='GeoJSON')

    return


def get_middle_hour(hours):
    """
    Calculate the middle hour between two given hours, handling day wraparound.

    Args:
        hours (list): List of two integers representing hours.

    Returns:
        int: Middle hour between the two input hours.
    """
    if len(hours) != 2:
        raise ValueError("The list must contain exactly two integers representing hours.")

    h1, h2 = hours
    if h1 > h2:
        h1, h2 = h2, h1

    direct_mid = (h1 + h2) // 2
    wrap_around_mid = (h1 + h2 + 24) // 2 % 24

    if h2 - h1 <= 12:
        return direct_mid
    else:
        return wrap_around_mid


def get_time_label(period_dict):
    """
    Generate time labels for each hour in a period dictionary.

    Args:
        period_dict (dict): Dictionary with start and end times as keys and labels as values.

    Returns:
        dict: Dictionary mapping time strings to labels.
    """
    from datetime import datetime, timedelta

    start_time = datetime.strptime(list(period_dict.keys())[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(list(period_dict.keys())[1], '%Y-%m-%d %H:%M:%S')

    start_count = list(period_dict.values())[0]

    num_hours = int((end_time - start_time).total_seconds() // 3600)

    result_dict = {}
    for i in range(num_hours + 1):
        current_time = start_time + timedelta(hours=i)
        current_count = start_count + i
        result_dict[current_time.strftime('%Y-%m-%d %H:%M:%S')] = current_count

    return result_dict


def merge_inundation_info_2_net(dir_net, dir_road, period_dict, period_split, table_name):
    """
    Merge inundation information into network data for different time periods.

    Args:
        dir_net (str): Path to network SQLite database.
        dir_road (str): Path to road data with inundation information.
        period_dict (dict): Period dictionary.
        period_split (dict): Period split dictionary.
        table_name (str): Name of the table in the database.

    Returns:
        dict: Dictionary with network data for each period.
    """
    import sqlite3

    conn = sqlite3.connect(dir_net)
    sim_net = pd.read_sql_query('SELECT * FROM ' + table_name, conn)
    sim_shp = gpd.read_file(dir_road)
    sim_net['A'] = sim_net['a']
    sim_net['B'] = sim_net['b']

    new_sim_net = {}
    labels = get_time_label(period_dict)
    for k, v in period_split.items():
        h = get_middle_hour(v)
        label = labels[f'2016-10-09 {h:02}:00:00']
        new_sim_net[k] = [
            label,
            sim_net.copy().merge(
                sim_shp[['A', 'B', f'max_depth_{label}', f'mean_depth_{label}']],
                on=['A', 'B'], how='left',
            ).drop_duplicates(
                subset=['A', 'B'], keep='first'
            ).drop(
                columns=['A', 'B']
            )
        ]

    return new_sim_net


def edit_net_using_inundation(net):
    """
    Edit network data using inundation information to adjust travel times.

    Args:
        net (dict): Network data dictionary.

    Returns:
        dict: Edited network data with adjusted travel times.
    """
    converter = InundationToSpeed(30)
    new_net = {}
    for k, v in net.items():
        l = v[0]
        df = v[1]
        df['dummy_speed'] = [1] * len(df)
        converter.cutoff(df[f'max_depth_{l}'])
        converter.reduce(df[f'mean_depth_{l}'], df['dummy_speed'], )
        df['dummy_speed_inundated'] = converter.apply_orig_speed(df['dummy_speed'], )

        # df = df[df['dummy_speed_inundated'] != 1e-05]
        df.loc[:, 'length'] = df['length'] / df['dummy_speed_inundated']
        df.loc[:, 'distance'] = df['distance'] / df['dummy_speed_inundated']

        df = df.drop(
            columns=['dummy_speed', 'dummy_speed_inundated'] + list(df.filter(regex='_depth_').columns)
        )
        new_net[k] = df

    return new_net


def edit_net_sqlite(original_sql, new_sql, nets, period_split, table_name):
    """
    Edit SQLite network databases for different time periods.

    Args:
        original_sql (str): Path to original SQLite database.
        new_sql (str): Directory to save new databases.
        nets (dict): Network data dictionary.
        period_split (dict): Period split dictionary.
        table_name (str): Name of the table to edit.
    """
    import os
    import shutil
    import sqlite3

    original_name = original_sql.split('/')[-1]
    for k, v in period_split.items():
        new_name = f'{k}_{original_name}'
        new_dir = f'{new_sql}/{new_name}'
        shutil.copy(original_sql, new_dir)
        df = nets[k]

        # create a new table with format
        conn = sqlite3.connect(new_dir)
        cursor = conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        command = cursor.fetchone()[0]
        command = command.replace(f'{table_name}', f'{table_name}__edited')
        cursor.execute(command)
        conn.commit()

        # delete the old one and insert new info with the old table name
        conn = sqlite3.connect(new_dir)
        conn.execute(f'DROP TABLE IF EXISTS {table_name}')
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.commit()

        # convert format to the new table
        conn = sqlite3.connect(new_dir)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name}__edited);")
        sql_col = cursor.fetchall()
        assert len(sql_col) == len(df.columns)

        command_head = f'INSERT INTO {table_name}__edited ('
        command_body = 'SELECT '
        for c, t in zip(df.columns, sql_col):
            command_head += f'{c}, '
            command_body += f'CAST({c} AS {t[2]}), '
        command_head = command_head[:-2] + ') '
        command_body = command_body[:-2] + ' '
        command = command_head + command_body + f'FROM {table_name};'
        cursor.execute(command)
        conn.commit()

        # delete wrong schema and rename the new one
        conn.execute(f'DROP TABLE IF EXISTS {table_name}')
        # cursor.execute(f"ALTER TABLE {table_name}__edited RENAME TO {table_name};")
        input('Renaming does not work because rtree is not properly configured for the geometry column.'
              'Using DB browser to rename manually. Continue?')

    return


def add_geo_unit(roads, dir_unit, id_col_geo):
    """
    Add geographic unit information to road segments.

    Args:
        roads (GeoDataFrame): Road segment data.
        dir_unit (str): Path to geographic unit boundary file.
        id_col_geo (list): List of column names for geographic identifiers.

    Returns:
        GeoDataFrame: Road segments with geographic unit information.
    """
    unit_geo = gpd.read_file(dir_unit)
    unit_geo = unit_geo[id_col_geo + ['geometry']]
    unit_geo = unit_geo.to_crs(roads.crs)
    roads_updated = roads.sjoin(unit_geo, how='left')
    return roads_updated


def merge_roads_demographic_bg(roads, demographic):
    """
    Merge demographic data with roads at the block group level.

    Args:
        roads (GeoDataFrame): Road segment data.
        demographic (DataFrame): Demographic data.

    Returns:
        GeoDataFrame: Road segments with demographic information.
    """
    roads['id'] = roads['COUNTYFP'] + roads['TRACTCE'] + roads['BLKGRPCE']
    assert (roads['COUNTYFP'].value_counts().iloc[0] / roads['COUNTYFP'].value_counts().sum()) > 0.99, \
        'One county must be dominating.'
    demographic['tract_name_adapt'] = '0' + (demographic['tract_name'].astype(float) * 100).astype(int).astype(str)
    demographic['id'] = (
            roads['COUNTYFP'].value_counts().index[0]
            + demographic['tract_name_adapt']
            + demographic['block_group_name']
    )
    roads = roads.merge(demographic, how='left', on='id')
    return roads


def calculate_severity_metric(group, cutoff_thr, depth_unit, depth_cols,):
    """
    Calculate severity metric for a group of road segments.

    Args:
        group (DataFrame): Group of road segments.
        cutoff_thr (float): Cutoff threshold for severity calculation.
        depth_unit (str): Unit of depth measurement ('cm' or 'ft').
        depth_cols (list): List of depth column names.

    Returns:
        tuple: (metric, income) - severity metric and average income.
    """
    group['ave_max_depth'] = group[depth_cols].mean(axis=1)
    max_severity = (group['length'] * cutoff_thr).sum()
    if depth_unit == 'cm':
        severity = (group['length'] * group['ave_max_depth']).sum()
    elif depth_unit == 'ft':
        severity = (group['length'] * group['ave_max_depth']*30.48).sum()
    else:
        raise ValueError('Specify unit.')
    metric = severity / max_severity
    income = group['demographic_value'].mean()
    return metric, income


def calculate_severity_metric_by_period(group, cutoff_thr, depth_unit, depth_cols, period_split, period_dict):
    """
    Calculate severity metrics for different time periods.

    Args:
        group (DataFrame): Group of road segments.
        cutoff_thr (float): Cutoff threshold for severity calculation.
        depth_unit (str): Unit of depth measurement ('cm' or 'ft').
        depth_cols (list): List of depth column names.
        period_split (dict): Period split dictionary.
        period_dict (dict): Period dictionary.

    Returns:
        tuple: (metric_list, income) - list of severity metrics and average income.
    """
    def if_in_period(hour):
        if start < end:
            if start <= hour < end:
                return True
        else:
            if hour >= start or hour < end:
                return True
        return False

    metric_list = []
    for period, (start, end) in period_split.items():
        depth_cols_updated = [
            i for i in depth_cols if if_in_period(
                int(i.split("_")[-1]) - list(period_dict.values())[0]
            )
        ]
        group_cols = group[depth_cols_updated]
        group_cols = group_cols.fillna(0)
        group['ave_max_depth'] = group_cols.mean(axis=1)
        max_severity = (group['length'] * cutoff_thr).sum()
        if depth_unit == 'cm':
            severity = (group['length'] * group['ave_max_depth']).sum()
        elif depth_unit == 'ft':
            severity = (group['length'] * group['ave_max_depth']*30.48).sum()
        else:
            raise ValueError('Specify unit.')
        metric = severity / max_severity
        metric_list.append(metric)
    return metric_list, group['demographic_value'].mean()


def calculate_congestion_metric(group, ff_time, congestion_time, manually_cut=10):
    """
    Calculate congestion metric for a group of road segments.

    Args:
        group (DataFrame): Group of road segments.
        ff_time (str): Column name for free-flow time.
        congestion_time (str): Column name for congestion time.
        manually_cut (float, optional): Manual cutoff value. Defaults to 10.

    Returns:
        tuple: (metric, income) - congestion metric and average income, or (None, None) if no valid data.
    """
    group = group[group[ff_time] > 1e-3]
    group = group[group[congestion_time] > 1e-3]
    group = group[group['LENGTH_full_net'] != 0]  # remove error
    group = group[group[ff_time] < manually_cut]  # remove roads that are cut off
    if len(group) == 0:
        return None, None
    metric = (group[congestion_time] - group[ff_time]).sum() / group[ff_time].sum()
    income = group['demographic_value'].mean()
    return metric, income


def get_congestion_metrics(dir_road, period_short, dir_road_cube6, dir_bg_boundaries, dir_income_bg):
    """
    Calculate congestion metrics for different time periods.

    Args:
        dir_road (str): Base directory for road data.
        period_short (list): List of period abbreviations.
        dir_road_cube6 (str): Path to CUBE6 road data.
        dir_bg_boundaries (str): Path to block group boundaries.
        dir_income_bg (str): Path to income data.

    Returns:
        list: List of DataFrames with congestion metrics for each period.
    """
    import pandas as pd
    import os
    from utils.preprocess_incidents import import_demographic
    metrics_df_list = []
    for l, n in zip(
            period_short, ['AM Peak', 'Midday', 'PM Peak', 'Night']
    ):
        if os.path.exists(f'{dir_road}_{l}/{l}_FDBKNET_LINK.dbf'):
            road_segment = merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road}_{l}/{l}_FDBKNET_LINK.dbf'
            )
            road_segment = add_geo_unit(road_segment, dir_bg_boundaries, ['COUNTYFP', 'TRACTCE', 'BLKGRPCE'])
            road_segment = merge_roads_demographic_bg(
                road_segment, import_demographic(dir_income_bg, ['B19013_001E'], ['9901'])
            )
            road_segment = road_segment[~road_segment['demographic_value'].isna()].fillna(0)
            congestion_metrics = road_segment.groupby('id').apply(
                lambda group: calculate_congestion_metric(group, 'FFTIME', 'TIME_1')
            )
            congestion_metrics_df = pd.DataFrame(
                congestion_metrics.tolist(), columns=['congestion', 'income'], index=congestion_metrics.index
            )
            congestion_metrics_df['period'] = [n] * len(congestion_metrics_df)
            metrics_df_list.append(congestion_metrics_df)
        else:
            raise ValueError('File missing.')

    return metrics_df_list

