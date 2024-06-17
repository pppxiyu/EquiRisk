import utils.preprocess as pp
import utils.preprocess_roads as pp_r
import utils.preprocess_incidents as pp_i
import utils.preprocess_graph as pp_g
import utils.modeling as mo

import geopandas as gpd
import matplotlib.pyplot as plt


def pull_road_data_osm():
    # pull road data
    pp_r.pull_roads_osm(
        'Virginia Beach, VA, USA', 'vb',
        './data/roads',
        '"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"',
    )


def pull_geocode_incident():
    # geocode incidents data ans save it
    pp_i.geocode_incident(
        './data/incidents/virginiaBeach_ambulance_timeData.csv',
        ['2013-01-01', '2013-01-02'],
        './data/incidents/geocoded/20130101-20130102.geojson',
    )


def save_inundated_roads():
    # overlapping the inundation and roads and save it
    road_segment = gpd.read_file(dir_road)
    for i in range(25, 48 + 1):
        road_segment = pp_r.add_water_depth_on_roads(
            road_segment,
            f'./data/inundation/tif_data/depth_objID_{i}.tif', i,
            {'max': 'max_depth', 'mean': 'mean_depth'}
        )
    road_segment.to_file(dir_road_inundated, driver='GeoJSON')


def save_rescue_data():
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
    rescue_station.to_csv(dir_rescue_station_n_nearest_geo)
    return


def build_full_graph_arcgis():
    import arcpy

    road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
    turn_restriction = pp_r.import_turn_restriction(dir_turn)

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
            2) Define travel cost attribute 'travel_time_s', unit is seconds. 
                Evaluator is field script !travel_time_s!.
            3) Define one-way restriction. Prohibit all against direction.
            4) Define turn restrictions. Add it in source. Apply all turn restrictions.
            5) Define inundation restriction. Keep it as default.
            6) Set costs and restrictions in the travel mode.
            7) Build network.
            Press Enter to continue.
        """
    )
    arcpy.BuildNetwork_na(f'{geodatabase_addr}/{fd_name}/{nd_name}')

    return


def calculate_all_routes():
    rescue_station = pp.import_rescue_station(dir_rescue_station_n_nearest_geo)
    incidents = pp_i.import_incident(dir_incidents)
    incidents = pp_i.add_actual_rescue_station(incidents, rescue_station)
    incidents = pp_i.add_nearest_rescue_station(incidents, rescue_station)
    incidents = pp_i.add_period_label(incidents, period_dict)
    road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)

    route_analysis = mo.RouteAnalysis(incidents, 'Number_nearest')
    route_analysis.run_route_analysis_arcgis(
        geodatabase_addr, fd_name, nd_name, nd_layer_name,
        rescue_station, road_segment,
    )


def calculate_incidents_with_gis_travel_time():
    incidents = pp_i.import_incident(dir_incidents)
    assert len(list(period_dict.values())) == 2
    incidents = pp_i.convert_feature_class_to_df(
        incidents,
        f'{geodatabase_addr}/route_results',
        ['normal'] + list(range(list(period_dict.values())[0], list(period_dict.values())[1] + 1))
    )
    incidents = pp_i.add_inaccessible_routes(incidents, dir_inaccessible_routes)
    return incidents


if __name__ == "__main__":

    crs_prj = 'epsg:32633'

    geodatabase_addr = './gis_analysis/arcgis_emergency_service_routing/arcgis_emergency_service_routing.gdb'
    fd_name = 'road_network'
    nd_name = 'road_nd'
    nd_layer_name = 'road_nd_layer'
    turn_name = 'turn_restriction'

    period_dict = {
            '2016-10-09 00:00:00': 25,
            '2016-10-09 23:00:00': 48,
        }
    speed_assigned = {
            'motorway': 55, 'motorway_link': 55, 'trunk': 55, 'trunk_link': 55,
            'primary': 55, 'primary_link': 55, 'secondary': 55, 'secondary_link': 55,
            'tertiary': 25, 'tertiary_link': 25, 'unclassified': 25, 'residential': 25, 'service': 25,
        }

    dir_rescue_station_n_nearest_geo = './data/rescue_team_location/rescue_stations_n_nearest_geo.csv'
    dir_incidents = 'data/incidents/geocoded/20160101-20161015.csv'
    dir_turn = './data/roads/turn_restriction_vb_overpass.geojson'
    dir_road = "./data/roads/road_segment_vb.geojson"
    dir_road_inundated = "./data/roads/road_segment_vb_inundated.geojson"
    dir_inaccessible_routes = "./data/incidents/inaccessible_route.json"

    # save_inundated_roads()
    # save_rescue_data()

    # build_full_graph_arcgis()
    # calculate_all_routes()

    # incidents = calculate_incidents_with_gis_travel_time()


    print()





