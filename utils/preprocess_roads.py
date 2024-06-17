import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Polygon
import warnings
import rasterio


def pull_roads_osm(city, city_abbr, folder, filter):
    import osmnx as ox
    import networkx as nx

    graph = ox.graph_from_place(
        city, custom_filter=f'["highway"~{filter}]',
        simplify=False, retain_all=True, truncate_by_edge=True
    )
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
    import networkx as nx

    gdf_nodes = gpd.read_file(f"{folder}/road_intersection_{city_abbr}.geojson")
    gdf_edges = gpd.read_file(f"{folder}/road_segment_{city_abbr}.geojson")
    graph = nx.read_gml(f'{folder}/graph_{city_abbr}.gml')
    return graph, gdf_edges, gdf_nodes


def add_roads_max_speed(gdf, speed, maxspeed_name):
    gdf[maxspeed_name] = gdf['highway'].map(speed)
    assert gdf['highway'].isna().any() == False, 'Speed was not assigned to all roads.'
    return gdf


def import_turn_restriction(addr):
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
    gdf = gpd.read_file(addr)
    gdf['id'] = gdf['id'].str.split('/').str[1]
    gdf = gdf[['id', 'highway']]
    return gdf


def add_travel_time_2_seg(roads, maxspeed_name, travel_time_name):
    # input('For add_travel_time_2_seg, input speed is mile/h, and output time is seconds.')
    roads['maxspeed_assigned_m_per_s'] = roads[maxspeed_name] * 1.60934 * 1000 / 60 / 60
    roads[travel_time_name] = roads['length'] / roads['maxspeed_assigned_m_per_s']
    return roads


def add_water_depth_on_roads(roads, inundation_tif_addr, label, new_col_name, remove_bridge=True):
    from rasterio.features import shapes

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
    warnings.warn('It is assumed that the no data value was set to a very big number by ArcGIS.')

    inundated_roads = roads.copy().sjoin(polygons, how='inner', predicate='intersects')

    inundated_roads_max = inundated_roads[['val']].groupby(inundated_roads.index).max()
    inundated_roads_mean = inundated_roads[['val']].groupby(inundated_roads.index).mean()

    roads[f"{new_col_name['max']}_{label}"] = inundated_roads_max['val']
    roads[f"{new_col_name['mean']}_{label}"] = inundated_roads_mean['val']

    if remove_bridge:
        roads.loc[roads['bridge'] == 'yes', f'{new_col_name["max"]}_{label}'] = np.nan
        roads.loc[roads['bridge'] == 'yes', f'{new_col_name["mean"]}_{label}'] = np.nan

    return roads


class InundationToSpeed:
    def __init__(self, thr=30, unit_checker=False):
        # input('threshold should be cm.')
        self.thr = thr
        self.unit_checker = unit_checker

        self.speed_unique = None
        self.curves = {}

        self.cut_series = None
        self.reduce_series = None
        return

    def build_decreasing_curve(self):
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
        return 0.0009 * series * series - 0.5529 * series + 86.9448

    def cutoff(self, depth_series):
        if self.unit_checker:
            input('For cutoff(), the input water depth should be ft. Enter to continue.')

        if_cut = depth_series * 30.48 >= self.thr
        self.cut_series = if_cut
        return if_cut

    def reduce(self, depth_series, maxspeed_series):
        if self.unit_checker:
            input('For reduce(), the input water depth should be ft, speed is mile/h. Enter to continue.')
        self.speed_unique = maxspeed_series.unique()

        speed_df = pd.DataFrame(index=maxspeed_series.index)
        speed_df['speed'] = [np.nan] * len(speed_df)

        self.build_decreasing_curve()

        maxspeed_series = maxspeed_series[(depth_series * 30.48 < self.thr) & (depth_series * 30.48 > 0)]
        depth_series = depth_series[(depth_series * 30.48 < self.thr) & (depth_series * 30.48 > 0)]
        speed_df['depth'] = depth_series

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
        if self.unit_checker:
            input('For apply_original_speed(), the input speed is mile/h. Enter to continue.')

        speed_series = self.cut_series.map({True: 1e-5, False: np.nan})
        speed_series = speed_series.fillna(self.reduce_series)
        speed_series = speed_series.fillna(original_maxspeed_series)

        return speed_series


def import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned):
    road_segment_inund = gpd.read_file(dir_road_inundated)
    road_segment_inund = add_roads_max_speed(
        road_segment_inund, speed_assigned,
        'maxspeed_assigned_mile'
    )

    road_segment_inund = add_travel_time_2_seg(
        road_segment_inund,
        'maxspeed_assigned_mile', 'travel_time_s'
    )

    converter = InundationToSpeed(30)
    for label in range(25, 48 + 1):
        converter.cutoff(
            road_segment_inund[f'max_depth_{label}']
        )
        converter.reduce(
            road_segment_inund[f'mean_depth_{label}'],
            road_segment_inund['maxspeed_assigned_mile'],
        )
        road_segment_inund[f'maxspeed_inundated_mile_{label}'] = converter.apply_orig_speed(
            road_segment_inund['maxspeed_assigned_mile'],
        )
        road_segment_inund = add_travel_time_2_seg(
            road_segment_inund,
            f'maxspeed_inundated_mile_{label}', f'travel_time_s_{label}'
        )
    return road_segment_inund

