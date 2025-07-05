import utils.preprocess_station as pp_s
import utils.preprocess_roads as pp_r
import utils.preprocess_incidents as pp_i
import utils.preprocess_graph as pp_g
import model as mo
from config_vb import *
import geopandas as gpd


def pull_road_data_osm():
    """
    Download and save OpenStreetMap road network data for Virginia Beach.
    
    Uses the OSMnx library to pull road data for the specified city and save it
    to the configured road folder. Filters for major road types including motorways,
    trunks, primary, secondary, tertiary, unclassified, residential, and service roads.
    
    Dependencies:
        - config_vb: city_name, dir_road_folder
        - utils.preprocess_roads: pull_roads_osm
    """
    pp_r.pull_roads_osm(
        city_name[0], city_name[1], dir_road_folder,
        '"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"',
    )


def pull_geocode_incident():
    """
    Geocode incident addresses and save the results.
    
    Processes raw incident data by geocoding addresses and saving the results
    to a GeoJSON file. This function is used for initial data preparation.
    
    Dependencies:
        - utils.preprocess_incidents: geocode_incident
    """
    # geocode incidents data ans save it
    pp_i.geocode_incident(
        './data/incidents/virginiaBeach_ambulance_timeData.csv',
        ['2013-01-01', '2013-01-02'],
        './data/incidents/geocoded/20130101-20130102.geojson',
    )


def save_inundated_roads():
    """
    Add water depth information to road segments and save the inundated road network.
    
    Processes road segments by adding water depth data from inundation raster files
    for each time period (25-48). The function handles bridge considerations and
    saves the result to the configured inundated road file.
    
    Dependencies:
        - config_vb: dir_road, dir_road_inundated
        - utils.preprocess_roads: add_water_depth_on_roads_w_bridge
    """
    # overlapping the inundation and roads and save it
    road_segment = gpd.read_file(dir_road)
    for i in range(25, 48 + 1):
        road_segment = pp_r.add_water_depth_on_roads_w_bridge(
            road_segment,
            f'./data/inundation/tif_data/depth_objID_{i}.tif', i,
            {'max': 'max_depth', 'mean': 'mean_depth'}
        )
    road_segment.to_file(dir_road_inundated, driver='GeoJSON')


def save_inundated_roads_4_sim():
    """
    Create inundated road network for simulation purposes.
    
    Similar to save_inundated_roads() but specifically for simulation data.
    Handles bridge removal and inundation under bridges differently than the
    main road network. Only creates the file if it doesn't already exist.
    
    Dependencies:
        - config_vb: dir_road, dir_road_cube6, dir_road_cube6_inundated
        - utils.preprocess_roads: add_water_depth_on_roads_w_bridge
    """
    import os
    if not os.path.exists(dir_road_cube6_inundated):
        road_segment = gpd.read_file(dir_road)
        road_shp_sim = gpd.read_file(dir_road_cube6)
        for i in range(25, 48 + 1):
            road_shp_sim = pp_r.add_water_depth_on_roads_w_bridge(
                road_shp_sim,
                f'./data/inundation/tif_data/depth_objID_{i}.tif', i,
                {'max': 'max_depth', 'mean': 'mean_depth'},
                remove_bridge=False,
                remove_inundation_under_bridge=True,
                geo_w_bridge=road_segment,
            )
        road_shp_sim.to_file(dir_road_cube6_inundated, driver='GeoJSON')


def save_rescue_data():
    """
    Process and save rescue station data with geographic information.
    
    Imports rescue station data and saves it to the configured location.
    This function prepares rescue station data for use in routing analysis.
    Note: Nearest segment and intersection calculations are commented out.
    
    Dependencies:
        - config_vb: dir_rescue_station, dir_rescue_station_n_nearest_geo
        - utils.preprocess_station: import_rescue_station
    """
    # process rescue station data and save it
    # road_intersection = gpd.read_file(f"data/VB/roads/road_intersection_vb.geojson")
    # road_segment = gpd.read_file(dir_road)
    rescue_station = pp_s.import_rescue_station(dir_rescue_station)
    # rescue_station['nearest_segment'] = rescue_station.to_crs(crs_prj).geometry.apply(
    #     lambda x: pp_s.add_nearest_segment(x, road_segment.to_crs(crs_prj))
    # )
    # rescue_station['nearest_intersection'] = rescue_station.to_crs(crs_prj).geometry.apply(
    #     lambda x: pp_s.add_nearest_intersection(x, road_intersection.to_crs(crs_prj))
    # )
    rescue_station.to_csv(dir_rescue_station_n_nearest_geo)
    return


def match_osm_n_vdot():
    """
    Match OpenStreetMap road data with VDOT (Virginia Department of Transportation) data.
    
    Creates a matching between OSM road segments and VDOT road network data.
    This matching is essential for incorporating VDOT traffic information into
    the OSM-based road network.
    
    Dependencies:
        - config_vb: dir_road_inundated, dir_road_cube6, dir_match_osm_n_VDOT, dir_road_cube6_out_c
        - utils.preprocess_roads: match_osm_n_VDOT
    """
    pp_r.match_osm_n_VDOT(
        dir_road_inundated, dir_road_cube6, dir_match_osm_n_VDOT,
        dir_VDOT_info=f'{dir_road_cube6_out_c}/AM_PK_FDBKNET_LINK.dbf',
    )


def build_full_graph_arcgis():
    """
    Build a complete ArcGIS network dataset with turn restrictions and travel modes.
    
    Creates a comprehensive network dataset in ArcGIS that includes:
    - Road segments with inundation information
    - Turn restrictions
    - Travel modes and cost attributes
    - Network topology
    
    This function requires manual intervention in ArcGIS to configure travel modes
    and network properties. It sets up the foundation for all routing analyses.
    
    Dependencies:
        - config_vb: geodatabase_addr, fd_name, nd_name, turn_name, dir_road_inundated, 
                     dir_turn, speed_assigned
        - utils.preprocess_roads: import_road_seg_w_inundation_info, import_turn_restriction
        - utils.preprocess_graph: build_feature_dataset_arcgis, build_network_dataset_arcgis, 
                                 add_turn_restriction_arcgis
        - model: RouteAnalysis, ServiceAreaAnalysis
    """
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


def calculate_all_routes(op):
    """
    Calculate routes for all incidents under different operational scenarios.
    
    This function implements different routing strategies for emergency response:
    - OP0: Normal conditions (baseline)
    - OP1: Nearest origin assignment
    - OP2: Real origin assignment (actual ambulance locations)
    - OP3: Everyday congestion conditions
    - OP4: Flooding congestion conditions
    - OP5: Combined congestion and origin shift
    
    Each option uses different combinations of:
    - Origin assignment methods (nearest vs. real)
    - Traffic conditions (normal vs. congested)
    - VDOT speed data integration
    
    Args:
        op (int): Operation mode (0-5) specifying the routing scenario.
        
    Dependencies:
        - config_vb: All directory paths and configuration parameters
        - utils.preprocess_station: import_rescue_station
        - utils.preprocess_incidents: import_incidents_add_info
        - utils.preprocess_roads: import_road_seg_w_inundation_info, merge_road_info_VDOT
        - model: RouteAnalysis
    """
    rescue_sta = pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo)
    icd = pp_i.import_incidents_add_info(
        dir_incidents, rescue_sta, period_dict, routing_nearest=dir_incidents_routing_nearest,
    )

    if op == 0:
        # OP0: normal
        road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
        route_analysis = mo.RouteAnalysis(icd, 'Number_nearest')
        route_analysis.run_route_analysis_arcgis(
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_sta, road_segment, if_do_normal=True, if_do_flood=False,
        )

    if op == 1:
        # OP1: nearest origin
        road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
        route_analysis = mo.RouteAnalysis(icd, 'Number_nearest')
        route_analysis.run_route_analysis_arcgis(
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_sta, road_segment, if_do_normal=False,
        )

    elif op == 2:
        # OP2: real origin
        road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
        route_analysis = mo.RouteAnalysis(icd, 'Rescue Squad Number', mode_label='_o')
        route_analysis.run_route_analysis_arcgis(
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_sta, road_segment,
        )

    elif op == 3:
        # OP3: everyday congestion
        road_segment_VDOT = {
            '9-13': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_c}/AM_PK_FDBKNET_LINK.dbf'
            ),
            '13-19': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_c}/Md_OP_FDBKNET_LINK.dbf'
            ),
            '19-22': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_c}/PM_PK_FDBKNET_LINK.dbf'
            ),
            '22-9': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_c}/Nt_OP_FDBKNET_LINK.dbf'
            ),
        }  # NOTE: UTC TIME !!! 2016-10-09 was in EDT.  Local time  + 4 hours = UTC time.
        road_segment = pp_r.import_road_seg_w_inundation_info(
            dir_road_inundated, speed_assigned,
            VDOT_speed=road_segment_VDOT,
            VDOT_speed_col = 'CSPD_1',
            osm_match_vdot=dir_match_osm_n_VDOT,
        )
        route_analysis = mo.RouteAnalysis(icd, 'Number_nearest', mode_label='_daily_c')
        route_analysis.run_route_analysis_arcgis(
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_sta, road_segment,
        )

    elif op == 4:
        # OP4: flooding congestion
        road_segment_VDOT = {
            '9-13': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_AM_PK/AM_PK_FDBKNET_LINK.dbf'
            ),
            '13-19': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_Md_OP/Md_OP_FDBKNET_LINK.dbf'
            ),
            '19-22': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_PM_PK/PM_PK_FDBKNET_LINK.dbf'
            ),
            '22-9': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_Nt_OP/Nt_OP_FDBKNET_LINK.dbf'
            ),
        }  # NOTE: UTC TIME !!! 2016-10-09 was in EDT.  Local time  + 4 hours = UTC time.
        road_segment = pp_r.import_road_seg_w_inundation_info(
            dir_road_inundated, speed_assigned,
            VDOT_speed=road_segment_VDOT,
            VDOT_speed_col='TIME_1',
            osm_match_vdot=dir_match_osm_n_VDOT,
        )
        route_analysis = mo.RouteAnalysis(icd, 'Number_nearest', mode_label='_flood_c')
        route_analysis.run_route_analysis_arcgis(
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_sta, road_segment,
        )

    elif op == 5:
        # OP5: congestion and origin shift
        road_segment_VDOT = {
            '9-13': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_AM_PK/AM_PK_FDBKNET_LINK.dbf'
            ),
            '13-19': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_Md_OP/Md_OP_FDBKNET_LINK.dbf'
            ),
            '19-22': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_PM_PK/PM_PK_FDBKNET_LINK.dbf'
            ),
            '22-9': pp_r.merge_road_info_VDOT(
                dir_road_cube6, f'{dir_road_cube6_out_d}_Nt_OP/Nt_OP_FDBKNET_LINK.dbf'
            ),
        }  # NOTE: UTC TIME !!! 2016-10-09 was in EDT.  Local time  + 4 hours = UTC time.
        road_segment = pp_r.import_road_seg_w_inundation_info(
            dir_road_inundated, speed_assigned,
            VDOT_speed=road_segment_VDOT,
            VDOT_speed_col='TIME_1',
            osm_match_vdot=dir_match_osm_n_VDOT,
        )
        route_analysis = mo.RouteAnalysis(icd, 'Rescue Squad Number', mode_label='_all')
        route_analysis.run_route_analysis_arcgis(
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_sta, road_segment,
        )


if __name__ == "__main__":

    """
    Modeling and data preparation:
        Before analysis, the approach in the Method section is implemented to prepare data 
        for statistical analysis. Developers can use cache data provided for the core analysis.
    """
    # pull_road_data_osm()  # save road data

    # save_inundated_roads()
    # save_rescue_data()
    # build_full_graph_arcgis()
    #
    # for i in [0, 1, 2, 3, 4, 5]:
    #     calculate_all_routes(op=i)

    # # run service area and save to arcgis
    # incidents = pp_i.import_incidents_add_info(
    #     dir_incidents,
    #     pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo),
    #     period_dict, routing_nearest=dir_incidents_routing_nearest,
    # )
    # road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
    # service_area_analysis = mo.ServiceAreaAnalysis(service_area_threshold, incidents, 'Number_nearest')
    # service_area_analysis.run_service_area_analysis_arcgis(
    #     geodatabase_addr, fd_name, nd_name, nd_layer_name,
    #     pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo),
    #     road_segment, if_do_normal=True, if_do_flood=True,
    # )

    #######
    # ###### Other data processing: edit the net in CUBE
    # save_inundated_roads_4_sim()
    #
    # nets = pp_r.merge_inundation_info_2_net(
    #     dir_road_cube7, dir_road_cube6_inundated, period_dict, period_split,
    #     'cubedb__Master_Network_CUBE7__link'
    # )
    # updated_nets = pp_r.edit_net_using_inundation(nets)
    # pp_r.edit_net_sqlite(
    #     dir_road_cube7, dir_road_cube7_inundated, updated_nets, period_split,
    #     'cubedb__Master_Network_CUBE7__link',
    # )

    print('End of the program.')
