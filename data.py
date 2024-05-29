import utils.preprocess as pp
import utils.preprocess_roads as pp_r
import utils.preprocess_incidents as pp_i
import utils.preprocess_graph as pp_g

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import xyzservices.providers as xyz
# from collections import Counter
# import plotly.figure_factory as ff

import arcpy


def pull_road_data_osm():
    # pull road data
    pp_r.pull_roads_osm(
        'Virginia Beach, VA, USA', 'vb',
        './data/roads',
        '"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"',
    )


def road_data():
    road_g, road_seg, road_inter = pp_r.import_roads_osm('./data/roads', 'vb')
    road_seg = pp_r.add_roads_max_speed(
        road_seg,
        {
            'motorway': 55, 'motorway_link': 55, 'trunk': 55, 'trunk_link': 55,
            'primary': 55, 'primary_link': 55, 'secondary': 55, 'secondary_link': 55,
            'tertiary': 25, 'tertiary_link': 25, 'unclassified': 25, 'residential': 25, 'service': 25,
        }
    )
    turn_restri = pp_r.import_turn_restriction('./data/roads/turn_restriction_vb_overpass.geojson')
    return road_g, road_seg, road_inter, turn_restri


def rescue_data():
    # process rescue station data and save it
    road_intersection = gpd.read_file(f"./data/roads/road_intersection_vb.geojson")
    road_segment = gpd.read_file(f"./data/roads/road_segment_vb.geojson")
    rescue_station = pp.import_rescue_station('./data/rescue_team_location/rescue_stations.txt')
    rescue_station['nearest_segment'] = rescue_station.to_crs(crs_prj).geometry.apply(
        lambda x: pp.add_nearest_segment(x, road_segment.to_crs(crs_prj))
    )
    rescue_station['nearest_intersection'] = rescue_station.to_crs(crs_prj).geometry.apply(
        lambda x: pp.add_nearest_intersection(x, road_intersection.to_crs(crs_prj))
    )
    rescue_station.to_csv('./data/rescue_team_location/rescue_stations_n_nearest_geo.csv')
    return


def geocode_incident_data():
    # geocode incidents data
    pp_i.geocode_incident(
        './data/ambulance/virginiaBeach_ambulance_timeData.csv',
        ['2013-01-01', '2013-01-02'],
        './data/ambulance/geocoded/20130101-20130102.csv',
    )


def build_full_graph():

    # param
    road_segment = gpd.read_file(f"./data/roads/road_segment_vb.geojson")
    geodatabase_addr = './gis_analysis/arcgis_emergency_service_routing/arcgis_emergency_service_routing.gdb'
    fd_name = 'road_network'
    nd_name = 'road_nd'
    nd_layer_name = 'road_nd_layer'
    turn_name = 'turn_restriction'
    max_edge_turn_fd = 8

    wrong_restriction_osm_list = [
        8281368, 8898632, 10393138, 10393444, 10393445,
        10393446, 11367492, 11367493, 11735305, 11735306,
        14109533
    ]
    wrong_restriction_osmnx_list = [  # eliminating is technically wrong, but does not affect results
        10064810, 14185206, 14594122, 17505468
    ]
    fields = [  # do not change the head, the sequence of elements are used below
        'SHAPE@',
        'Edge1End',
        'Edge1FCID', 'Edge1FID', 'Edge1Pos',
        'Edge2FCID', 'Edge2FID', 'Edge2Pos',
        'Edge3FCID', 'Edge3FID', 'Edge3Pos',
        'Edge4FCID', 'Edge4FID', 'Edge4Pos',
        'Edge5FCID', 'Edge5FID', 'Edge5Pos',
        'Edge6FCID', 'Edge6FID', 'Edge6Pos',
        'Edge7FCID', 'Edge7FID', 'Edge7Pos',
        'Edge8FCID', 'Edge8FID', 'Edge8Pos',
    ]
    assert ((len(fields) - 2) / 3) == max_edge_turn_fd, 'Field length is wrong.'

    # import data
    road_segment = pp_r.add_roads_max_speed(
        road_segment,
        {
            'motorway': 55, 'motorway_link': 55, 'trunk': 55, 'trunk_link': 55,
            'primary': 55, 'primary_link': 55, 'secondary': 55, 'secondary_link': 55,
            'tertiary': 25, 'tertiary_link': 25, 'unclassified': 25, 'residential': 25, 'service': 25,
        }
    )
    road_segment = pp_r.add_travel_time_2_seg(road_segment)
    turn_restriction = pp_r.import_turn_restriction('./data/roads/turn_restriction_vb_overpass.geojson')

    # create network dataset
    if arcpy.Exists(f'{geodatabase_addr}/{fd_name}/{nd_name}') is not True:
        pp_g.build_feature_dataset_arcgis(geodatabase_addr, road_segment)
        pp_g.build_network_dataset_arcgis(geodatabase_addr)
    desc = arcpy.Describe(f'{geodatabase_addr}/{fd_name}/{nd_name}')
    if hasattr(desc, 'turnFeatureClasses') is not True:
        arcpy.na.CreateTurnFeatureClass(
            f'{geodatabase_addr}/{fd_name}',
            turn_name, 8,
        )
        pp_g.add_turn_restriction_arcgis(
            f'{geodatabase_addr}/{fd_name}',
            fields,
            turn_name,
            turn_restriction, road_segment,
            wrong_restriction_osm_list + wrong_restriction_osmnx_list,
        )

    # manual input
    input(
        """
            Must conduct following operation in network dataset properties manually:
            1) Create travel mode, name it DriveTime
            2) Define travel cost ('travel_time_s')
            3) Define one-way restriction. Prohibit all against direction.
            4) Define turn restrictions manually. Apply all turn restrictions.
            5) Set cost in travel modes.
            Confirm? Press Enter to continue.
        """
    )
    arcpy.BuildNetwork_na(f'{geodatabase_addr}/{fd_name}/{nd_name}')
    return


if __name__ == "__main__":
    crs_geo = 'epsg:4326'
    crs_prj = 'epsg:32633'



####################################################
    roads = pp.getWaterDepthOnRoads(
        roads,
        './data/inundation/tifData/depth_objID_35.tif',
        './data/inundation/croppedByRoads/croppedByRoads.tif'
    )

    roads.drop(['line', 'midpoint','buffers','buffersUnscaled'], axis=1).to_file(
        './data/roads/savedInundatedRoads/roads_with_objID_35.shp'
    )
    roads['line'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_line.shp')
    roads['midpoint'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_midpoint.shp')
    roads['buffers'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_buffers.shp')
    roads['buffersUnscaled'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_buffersUnscaled.shp')

    # # consider bridges in road network (PENDING)
    # bridges = gpd.read_file('./data/roads/bridges/bridgePolygon.shp').to_crs(str(inundation.crs))
    # inundation_cropped = inundationCutter(inundation, bridges, True, True)


