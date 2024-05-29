import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Polygon


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


def add_roads_max_speed(gdf, speed):
    gdf['maxspeed_assigned_mile'] = gdf['highway'].map(speed)
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


def add_travel_time_2_seg(roads):
    # roads_crs = roads.crs
    # roads = roads.to_crs('epsg:32633')
    # roads['length'] = roads.geometry.length
    roads['maxspeed_assigned_m_per_s'] = roads['maxspeed_assigned_mile'] * 1.60934 * 1000 / 60 / 60
    roads['travel_time_s'] = roads['length'] / roads['maxspeed_assigned_m_per_s']
    # roads = roads.to_crs(roads_crs)
    return roads


def legacy__moveDuplicates(joined):
    # move duplicates, keep the row with higher width
    u, c = np.unique(joined.OBJECTID_left.values, return_counts=True)
    duplicates = u[c > 1]
    joined_noDuplicates = joined.copy()
    for dup in duplicates:
        du = joined[joined.OBJECTID_left == dup]
        joined_noDuplicates = joined_noDuplicates[joined_noDuplicates.OBJECTID_left != dup]
        duOne = du[du.aveWidth == du.aveWidth.max()]
        joined_noDuplicates = pd.concat([joined_noDuplicates, duOne])
    return joined_noDuplicates.sort_values(by=['OBJECTID_left'])


def legacy__createSurface4roads(roads, roadSurfaces):
    # USE: create a geoDataFrame containing the column of average width and full polygon (might include multiple road segments) for each road
    # spatial join road lines and surfaces
    if roads.crs != roadSurfaces.crs:
        return 'crs not consistent'
    roadSurfaces['aveWidth'] = roadSurfaces.Shapearea / roadSurfaces.Shapelen
    roads['midpoint'] = roads.geometry.interpolate(0.5, normalized=True)
    roads = roads.set_geometry("midpoint", crs=roadSurfaces.crs)
    roads = roads.rename(columns={"geometry": "line"})
    joined = roads.sjoin(roadSurfaces, how="left", predicate='within')
    # move duplicates/nan
    joined_updated = legacy__moveDuplicates(joined)
    joined_updated.loc[np.isnan(joined_updated.aveWidth), [
        'aveWidth']] = joined_updated.aveWidth.mean()  # assign width to missing roads
    # attach roadSurface polygons
    joined_updated['OBJECTID_right'] = joined_updated.OBJECTID_right.astype('Int64')
    roadSurfaces_temp = roadSurfaces[['OBJECTID', 'geometry']].rename(
        {'OBJECTID': 'OBJECTID_right', 'geometry': 'surfacePolygon'}, axis=1)
    roadSurfaces_temp.loc[len(roadSurfaces_temp)] = [np.nan, Polygon()]
    roadSurfaces_temp.OBJECTID_right = roadSurfaces_temp.OBJECTID_right.astype('Int64')
    joined_updated = joined_updated.merge(roadSurfaces_temp, how='left', on='OBJECTID_right')
    joined_updated = joined_updated.set_geometry('surfacePolygon').set_crs(roadSurfaces.crs)
    return joined_updated


def legacy_make_surface_4_lines(roads, surfaceAddress, scale=2.7):
    roadSurfaces = gpd.read_file(surfaceAddress)
    surfaces4roads = legacy__createSurface4roads(roads, roadSurfaces)

    roads['aveWidth'] = surfaces4roads.aveWidth
    roads['scaledRadius'] = roads['aveWidth'] / 2 * scale
    roads['buffers'] = roads.geometry.buffer(roads['scaledRadius'])
    roads['buffersUnscaled'] = roads.geometry.buffer(
        roads['aveWidth'] / 2 * 1.5)  # may be some errors in raw data, roads look good when multiply by 1.5
    roads = roads.rename(columns={"geometry": "line"})
    roads = roads.set_geometry('buffers', crs=roadSurfaces.crs)

    roads['surface'] = [road.intersection(surface) if not road.intersection(surface).is_empty else roadUnscaled \
                        for road, surface, roadUnscaled in
                        zip(roads.geometry, surfaces4roads.geometry, roads.buffersUnscaled)]
    roads = roads.set_geometry('surface', crs=roadSurfaces.crs)
    return roads


def legacy_read_roads(roadAddress):
    roads = gpd.read_file(roadAddress)
    roads = roads.loc[-roads['geometry'].duplicated(), :]
    roads['OBJECTID'] = list(range(1, len(roads) + 1))
    roads = roads.reset_index(drop=True)
    return roads


def legacy_reload_roads(addr):
    addrSplit = addr.split('.shp')
    roads = gpd.read_file(addr)
    roads['line'] = gpd.read_file(addrSplit[0] + '_line.shp').geometry
    roads['midpoint'] = gpd.read_file(addrSplit[0] + '_midpoint.shp').geometry
    roads['buffers'] = gpd.read_file(addrSplit[0] + '_buffers.shp').geometry
    roads['buffersUnscaled'] = gpd.read_file(addrSplit[0] + '_buffersUnscaled.shp').geometry
    roads = roads.rename(columns={'geometry': 'surface', 'scaledRadi': 'scaledRadius'})
    roads = roads.set_geometry('surface')
    return roads
