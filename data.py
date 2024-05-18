import utils.preprocess as pp
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from mpl_toolkits.axes_grid1 import make_axes_locatable
import xyzservices.providers as xyz
from collections import Counter
import plotly.figure_factory as ff


if __name__ == "__main__":

    def incidents_preprocess():
        data = pp.incidents_add_time_addr('./data/ambulance/virginiaBeach_ambulance_timeData.csv')
        data = pp.incidents_add_origin(data, './data/rescue_team_location/rescueStations.txt')
        data = pp.incidents_geo_coding(
            data.loc['2013-01-01': '2013-12-31', :],
            './data/ambulance/geocoded_saved/20130101-20131231.csv',
        )


    def pull_roads_osm(city, city_abbr, folder):
        import osmnx as ox
        import networkx as nx

        road_type_to_get = """
            ["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|
            motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|
            busway|service|"]
        """
        graph = ox.graph_from_place(
            city,
            custom_filter=road_type_to_get,
            simplify=False, retain_all=True, truncate_by_edge=True
        )
        graph = ox.simplify_graph(graph, edge_attrs_differ=['osmid', 'oneway'])
        for u, v, data in graph.edges(data=True):
            if 'geometry' in data.keys():
                del data['geometry']
        gdf_nodes = ox.graph_to_gdfs(graph, edges=True)[0]
        gdf_edges = ox.graph_to_gdfs(graph, edges=True)[1]
        nx.write_gml(graph, f'{folder}/graph_{city_abbr}.gml')
        gdf_nodes.to_file(f"{folder}/road_intersection_{city_abbr}.geojson", driver="GeoJSON")
        gdf_edges.to_file(f"{folder}/road_segment_{city_abbr}.geojson", driver="GeoJSON")


    def import_roads_osm(folder, city_abbr):
        import networkx as nx
        import geopandas as gpd

        gdf_nodes = gpd.read_file(f"{folder}/road_intersection_{city_abbr}.geojson")
        gdf_edges = gpd.read_file(f"{folder}/road_segment_{city_abbr}.geojson")
        graph = nx.read_gml(f'{folder}/graph_{city_abbr}.gml')
        return graph, gdf_edges, gdf_nodes


    def import_turn_restriction(addr):
        import ast

        gdf = gpd.read_file(addr)
        gdf = gdf[['@id', '@relations']]
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

            for _, via_row in group[group['@role'] == 'via'].iterrows():
                row = [
                    rel,
                    group[group['@role'] == 'from']['@id'].iloc[0],
                    via_row['@id'],
                    group[group['@role'] == 'to']['@id'].iloc[0],
                ]
                row_list.append(row)
        gdf_out = pd.DataFrame(row_list, columns=['rel', 'from', 'via', 'to'])

        return gdf_out


    def import_bridge_tunnel(addr):
        gdf = gpd.read_file(addr)
        gdf['id'] = gdf['id'].str.split('/').str[1]
        gdf = gdf[['id', 'highway']]
        return gdf


    # pull_roads_osm('Virginia Beach, VA, USA', 'vb', './data/roads')
    road_graph, road_segment, road_intersection = import_roads_osm('./data/roads', 'vb')
    # turn_restriction = import_turn_restriction('./data/roads/turn_restriction_vb.geojson')

    roads = pp.readRoads('./data/roads/Streets.shp')
    roads = pp.makeSurface4Lines(roads, './data/roads/Road_Surfaces.shp', scale = 2.7)
    roads = pp.getWaterDepthOnRoads(roads, './data/inundation/tifData/depth_objID_35.tif', './data/inundation/croppedByRoads/croppedByRoads.tif')

    roads.drop(['line', 'midpoint','buffers','buffersUnscaled'], axis=1).to_file('./data/roads/savedInundatedRoads/roads_with_objID_35.shp')
    roads['line'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_line.shp')
    roads['midpoint'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_midpoint.shp')
    roads['buffers'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_buffers.shp')
    roads['buffersUnscaled'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_buffersUnscaled.shp')

    # # consider bridges in road network (PENDING)
    # bridges = gpd.read_file('./data/roads/bridges/bridgePolygon.shp').to_crs(str(inundation.crs))
    # inundation_cropped = inundationCutter(inundation, bridges, True, True)


