import utils.preprocess_station as pp_s
import utils.preprocess_roads as pp_r
import utils.preprocess_incidents as pp_i
import utils.preprocess_graph as pp_g
import model as mo
import utils.visualization as vis
import utils.regression as reg
from config_vb import *
# from config_chs import *
import geopandas as gpd
import os


def pull_road_data_osm():
    pp_r.pull_roads_osm(
        city_name[0], city_name[1], dir_road_folder,
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
        road_segment = pp_r.add_water_depth_on_roads_w_bridge(
            road_segment,
            f'./data/inundation/tif_data/depth_objID_{i}.tif', i,
            {'max': 'max_depth', 'mean': 'mean_depth'}
        )
    road_segment.to_file(dir_road_inundated, driver='GeoJSON')


def save_inundated_roads_4_sim():
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
    pp_r.match_osm_n_VDOT(
        dir_road_inundated, dir_road_cube6, dir_match_osm_n_VDOT,
        dir_VDOT_info=f'{dir_road_cube6_out_c}/AM_PK_FDBKNET_LINK.dbf',
    )


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


def calculate_all_routes(op):
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


def calculate_incidents_with_gis_travel_time(op):
    icd = pp_i.import_incident(dir_incidents)
    assert len(list(period_dict.values())) == 2

    if op == 1:
        # OP1: nearest origin
        icd = pp_i.convert_feature_class_to_df(
            icd,
            f'{geodatabase_addr}/route_results',
            ['normal'] + list(range(list(period_dict.values())[0], list(period_dict.values())[1] + 1)),
            mode_label='',
        )
        icd = pp_i.add_inaccessible_routes(
            icd, f'{dir_inaccessible_routes}.json'
        )

    if op == 2:
        # OP2: real origin
        icd = pp_i.convert_feature_class_to_df(
            icd,
            f'{geodatabase_addr}/route_results',
            ['normal'] + list(range(list(period_dict.values())[0], list(period_dict.values())[1] + 1)),
            mode_label='_o',
        )
        icd = pp_i.add_inaccessible_routes(
            icd, f'{dir_inaccessible_routes}_o.json'
        )

    if op == 3:
        # OP3: daily congestion
        icd = pp_i.convert_feature_class_to_df(
            icd,
            f'{geodatabase_addr}/route_results',
            list(range(list(period_dict.values())[0], list(period_dict.values())[1] + 1)),
            mode_label='_daily_c',
        )
        icd = pp_i.add_inaccessible_routes(
            icd, f'{dir_inaccessible_routes}_daily_c.json'
        )

    if op == 4:
        # OP4: flood congestion
        icd = pp_i.convert_feature_class_to_df(
            icd,
            f'{geodatabase_addr}/route_results',
            list(range(list(period_dict.values())[0], list(period_dict.values())[1] + 1)),
            mode_label='_flood_c',
        )
        icd = pp_i.add_inaccessible_routes(
            icd, f'{dir_inaccessible_routes}_flood_c.json'
        )

    if op == 5:
        # OP5:
        icd = pp_i.convert_feature_class_to_df(
            icd,
            f'{geodatabase_addr}/route_results',
            list(range(list(period_dict.values())[0], list(period_dict.values())[1] + 1)),
            mode_label='_all',
        )
        icd = pp_i.add_inaccessible_routes(
            icd, f'{dir_inaccessible_routes}_all.json'
        )

    return icd


def calculate_incidents_metrics(inc, demo_label='income'):
    # clean
    icd_c = inc.copy()
    icd_c = pp_i.add_period_label(icd_c, 'Call Date and Time', period_dict)
    icd_c = pp_i.convert_timedelta_2_seconds(icd_c, ['TravelTime', 'ResponseTime'])
    icd_c = icd_c[~icd_c['Total_Seconds'].isna()]
    icd_c = icd_c[~icd_c['TravelTime'].isna()]
    icd_c = icd_c[(icd_c['TravelTime'] < 0) | (icd_c['TravelTime'] >= 60)]

    # deal with infinity
    non_inf_over = icd_c[
        (icd_c['period_label'] != '') &
        (icd_c['Total_Seconds'] > icd_c['TravelTime'])
        ]
    non_inf_over.loc[:, ['over_rate']] = non_inf_over['Total_Seconds'] / non_inf_over['TravelTime']
    icd_c.loc[
        icd_c['Total_Seconds'] == -999, ['Total_Seconds']
    ] = icd_c.loc[icd_c['Total_Seconds'] == -999, 'TravelTime'] * non_inf_over['over_rate'].median()

    # numerical diff
    icd_c['diff_travel'] = icd_c['Total_Seconds'] - icd_c['TravelTime']
    icd_c = icd_c[~icd_c['diff_travel'].isna()]

    # categorical wellness
    icd_c['false_positive'] = ((icd_c['Total_Seconds'] <= service_area_threshold)
                                     & (icd_c['TravelTime'] > service_area_threshold))
    icd_c['false_negative'] = ((icd_c['Total_Seconds'] > service_area_threshold)
                                     & (icd_c['TravelTime'] <= service_area_threshold))
    icd_c['wellness'] = [0] * len(icd_c)
    icd_c.loc[icd_c['false_positive'] == True, 'wellness'] = -1
    icd_c.loc[icd_c['false_negative'] == True, 'wellness'] = 1
    icd_c = icd_c[~icd_c['wellness'].isna()]

    # add demo info - block groups
    icd_c = pp_i.add_geo_unit(icd_c, dir_bg_boundaries, ['TRACTCE', 'BLKGRPCE'])
    if demo_label == 'income':
        demo = pp_i.import_demographic(dir_income_bg, ['B19013_001E'],  ['9901'])
    elif demo_label == 'edu':
        demo = pp_i.import_demographic(dir_edu_bg, ['above_bch_ratio'],['9901'])
    elif demo_label == 'minority':
        demo = pp_i.import_demographic(dir_minority_bg, ['white_ratio'], ['9901'],)
    elif demo_label == 'age':
        demo = pp_i.import_demographic(dir_age_bg, ['5_to_65_ratio'], ['9901'])
    else:
        raise ValueError('Demographic information is missing.')
    icd_c = pp_i.merge_incidents_demographic_bg(icd_c, demo)
    icd_c = icd_c[~icd_c['demographic_value'].isna()]

    # split
    icd_f = icd_c[icd_c['period_label'] != '']
    icd_n = icd_c[icd_c['period_label'] == '']
    # pp_i.save_ict_to_shp(icd_f, dir_save_travel_time_est)

    # demo w geo
    demo = pp_i.merge_demographic_geo_bg(demo, dir_bg_boundaries)

    # aggr to geo units
    g_units_f = pp_i.aggr_incidents_geo(icd_f, period_dict, dir_bg_boundaries)
    g_units_n = pp_i.aggr_incidents_geo(icd_n, period_dict, dir_bg_boundaries)

    # outliers
    g_units_f = pp_i.delete_outlier_mahalanobis(g_units_f, ['demographic_value', 'diff_travel'])
    g_units_n = pp_i.delete_outlier_mahalanobis(g_units_n, ['demographic_value', 'diff_travel'])

    return icd_f, icd_n, demo, g_units_f, g_units_n


if __name__ == "__main__":

    ########
    # pull_road_data_osm()

    # save_inundated_roads()
    # save_rescue_data()
    # build_full_graph_arcgis()

    # for i in [0, 1, 2, 3, 4, 5]:
    #     calculate_all_routes(op=i)

    ###### Results 1
    # # inequality analysis
    # for x, s in zip([
    #         'Median household income (USD)',
    #         'Percent Bachelor\'s degree and higher',
    #         'Percent non-minority',
    #         'Percent population aged 5 to 65',
    #     ],
    #         ['income', 'edu', 'minority', 'age',]
    # ):
    #     _, _, _, geo_units_f_op1, geo_units_n_op1 = calculate_incidents_metrics(
    #         calculate_incidents_with_gis_travel_time(op=1), demo_label=s,
    #     )
    #     reg_model = reg.reg_spatial_lag(
    #         geo_units_f_op1, w_lag=1, method='ML', weight_method='Queen',  # spillover=True
    #     )
    #     vis.scatter_demo_vs_error(
    #         geo_units_f_op1,  color='#235689',  # flooding day
    #         reg_line=[reg_model.betas[1, 0], reg_model.betas[0, 0]],
    #         xaxis=x, save_label=s,
    #     )
    #     vis.scatter_demo_vs_error(
    #         geo_units_n_op1,  color='#42BCB2',  # normal period
    #         xaxis=x, save_label=s,
    #     )

    # # wellness calculation
    # incident_f, _, _, _, _ = calculate_incidents_metrics(
    #     calculate_incidents_with_gis_travel_time(op=1), demo_label='income',
    # )
    # incident_f = pp_i.add_econ_class(
    #     incident_f, middle_class_bar, 'demographic_value', ['wellness'],
    #     './data/VB/incidents/flood_ict_with_econ_class/flood_ict_with_econ_class.shp'
    # )
    # lower_class = incident_f[incident_f['demographic_value'] < middle_class_bar]['wellness'].mean()
    # print(f"Lower class ave wellness: {lower_class}")
    # middle_higher_class = incident_f[incident_f['demographic_value'] >= middle_class_bar]['wellness'].mean()
    # print(f"Middle and upper class ave wellness: {middle_higher_class}")
    # vis.bar_wellness(lower_class, middle_higher_class)

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

    # # travel time increase scatter and dist
    # vis.scatter_dist_icd_travel_time(pp_i.import_incident(dir_incidents), period_dict, 'scatter')
    # #######

    #### Regression analysis 2-5
    # _, _, _, geo_units_f_op1, _ = calculate_incidents_metrics(
    #     calculate_incidents_with_gis_travel_time(op=1), demo_label='income',
    # )
    # for option in [2, 3, 4, 5]:  # 2-origin; 3-daily traffic; 4-flood congestion; 5-all
    #     print(f'Analysis for Option {option}.')
    #     _, _, _, geo_units_f_op2345, _ = calculate_incidents_metrics(
    #         calculate_incidents_with_gis_travel_time(op=option), demo_label='income',
    #     )
    #     reg_model = reg.reg_spatial_lag(geo_units_f_op2345, w_lag=1, method='ML', weight_method='Queen',)
    #     reg.reg_shift_test_bootstrapping(
    #         geo_units_f_op1, geo_units_f_op2345,'ML', weight_method='Queen', w_lag=1,
    #         n_iter=5000,
    #     )
    #     vis.scatter_demo_vs_error(
    #         geo_units_f_op2345, color='#235689',
    #         reg_line=[reg_model.betas[1, 0], reg_model.betas[0, 0]],
    #         xaxis='Median household income (USD)', save_label='income',
    #         op_label=f'_op{option}',
    #     )

    #### Results 2
    # # microscale examination
    # rescue_station = pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo)
    # incidents = pp_i.import_incidents_add_info(
    #     dir_incidents, rescue_station, period_dict, routing_nearest=dir_incidents_routing_nearest,
    # )
    # incidents_shift = incidents[incidents['Number_nearest'] != incidents['Number_actual']]
    # incidents_shift_f = incidents_shift[~incidents_shift['period_actual'].isna()]
    # incidents_shift_f.loc[:, ['if_nearest_occupied', 'if_nearest_closed']] = incidents_shift_f.apply(
    #     lambda x: pp_s.check_closure_n_occupation(
    #         x, station_col='Number_nearest', time_col='Call Date and Time', incidents=incidents,
    #     ),
    #     axis=1, result_type='expand'
    # ).values
    # vis.map_origin_shift(incidents_shift_f, rescue_station, mode='actual')
    # vis.map_origin_shift(incidents_shift_f, rescue_station, mode='nearest')

    # # income vs service count
    # _, incidents_n, _, _, _ = calculate_incidents_metrics(
    #     calculate_incidents_with_gis_travel_time(op=1), demo_label='income',
    # )
    # vis.scatter_income_service_volumn(incidents_n, incidents_shift_f, 'scatter')
    # vis.scatter_income_service_volumn(incidents_n, incidents_shift_f, 'dist')

    # # income vs flooding severity
    # road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
    # road_segment = pp_r.add_geo_unit(road_segment, dir_bg_boundaries, ['TRACTCE', 'BLKGRPCE'])
    # road_segment = pp_r.merge_roads_demographic_bg(
    #     road_segment, pp_i.import_demographic(dir_income_bg, ['B19013_001E'],  ['9901'])
    # )
    # road_segment = road_segment[road_segment['bridge'].isna()]
    # road_segment = road_segment[~road_segment['demographic_value'].isna()].fillna(0)
    # depth_cols = [i for i in road_segment.columns if i.startswith('max_depth')]
    # road_segment[depth_cols] = (road_segment[depth_cols] * 30.48).clip(upper=road_cutoff_threshold)
    # severity_metrics = road_segment.groupby('id').apply(
    #     lambda group: pp_r.calculate_severity_metric(group, road_cutoff_threshold, 'cm', depth_cols)
    # )
    # import pandas as pd
    # vis.scatter_inundation_severity_vs_income(
    #     pd.DataFrame(
    #         severity_metrics.tolist(), columns=['severity', 'income'], index=severity_metrics.index
    #     )
    # )

    # # percent non-nearest
    # rescue_station = pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo)
    # incidents = pp_i.import_incidents_add_info(
    #     dir_incidents, rescue_station, period_dict, routing_nearest=dir_incidents_routing_nearest,
    # )
    # percent_normal = len(
    #     incidents[(incidents['period_actual'].isna()) & (incidents['Number_nearest'] != incidents['Number_actual'])]
    # ) / len(incidents[incidents['period_actual'].isna()])
    # percent_flood = len(
    #     incidents[(~incidents['period_actual'].isna()) & (incidents['Number_nearest'] != incidents['Number_actual'])]
    # ) / len(incidents[~incidents['period_actual'].isna()])
    # vis.bar_per_non_nearest(percent_normal, percent_flood)

    # # non-nearest reasons
    # rescue_station = pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo)
    # incidents = pp_i.import_incidents_add_info(
    #     dir_incidents, rescue_station, period_dict, routing_nearest=dir_incidents_routing_nearest,
    # )
    # incidents_shift = incidents[incidents['Number_nearest'] != incidents['Number_actual']]
    # incidents_shift.loc[:, ['if_nearest_occupied', 'if_nearest_closed']] = incidents_shift.apply(
    #     lambda x: pp_s.check_closure_n_occupation(
    #         x, station_col='Number_nearest', time_col='Call Date and Time', incidents=incidents,
    #     ),
    #     axis=1, result_type='expand'
    # ).values
    # vis.bar_per_nearest_reason(
    #     incidents_shift[incidents_shift['period_actual'].isna()],
    #     incidents_shift[~incidents_shift['period_actual'].isna()],
    # )

    # # ave income of normal and disrupted stations
    # rescue_station = pp_s.import_rescue_station(dir_rescue_station_n_nearest_geo)
    # incidents = pp_i.import_incidents_add_info(
    #     dir_incidents, rescue_station, period_dict, routing_nearest=dir_incidents_routing_nearest,
    # )
    # incidents_f, incidents_n, _, _, _ = calculate_incidents_metrics(
    #     calculate_incidents_with_gis_travel_time(op=1), demo_label='income',
    # )
    # incidents_n = incidents_n.merge(
    #     incidents[['incident_id', 'Number_actual', 'Number_nearest']], how='left', on='incident_id'
    # )
    # incidents_f = incidents_f.merge(
    #     incidents[['incident_id', 'Number_actual', 'Number_nearest']], how='left', on='incident_id'
    # )
    # vis.bar_ave_income_normal_disrupted_icd(
    #     incidents_n[incidents_n['Number_nearest'] == incidents_n['Number_actual']]['demographic_value'].mean(),
    #     incidents_n[incidents_n['Number_nearest'] != incidents_n['Number_actual']]['demographic_value'].mean(),
    #     incidents_f[incidents_f['Number_nearest'] == incidents_f['Number_actual']]['demographic_value'].mean(),
    #     incidents_f[incidents_f['Number_nearest'] != incidents_f['Number_actual']]['demographic_value'].mean(),
    # )
    #####

    ###### Results 3
    # # ambulance travel speed variation
    # _, incidents_n, _, _, _ = calculate_incidents_metrics(
    #     calculate_incidents_with_gis_travel_time(op=1), demo_label='income',
    # )
    # vis.line_hotspot_ave_time(*pp_i.get_hotspots_ave_time(incidents_n))

    # # complete net
    # for l in period_short:
    #     if os.path.exists(f'{dir_road_cube6_out_c}_{l}/{l}_FDBKNET_LINK.dbf'):
    #         road_segment = pp_r.merge_road_info_VDOT(
    #             dir_road_cube6, f'{dir_road_cube6_out_c}_{l}/{l}_FDBKNET_LINK.dbf'
    #         )
    #         vis.map_road_speed(road_segment, 'TIME_1', label=f'_complete_{l}')

    # # disrupted net
    # for l in period_short:
    #     if os.path.exists(f'{dir_road_cube6_out_d}_{l}/{l}_FDBKNET_LINK.dbf'):
    #         road_segment = pp_r.merge_road_info_VDOT(
    #             dir_road_cube6, f'{dir_road_cube6_out_d}_{l}/{l}_FDBKNET_LINK.dbf'
    #         )
    #         vis.map_road_speed(road_segment, 'TIME_1', label=f'_disrupted_{l}')

    # # income vs congestion
    # metrics_df_list_d = pp_r.get_congestion_metrics(
    #     dir_road_cube6_out_d, period_short, dir_road_cube6, dir_bg_boundaries, dir_income_bg
    # )
    # metrics_df_list_c = pp_r.get_congestion_metrics(
    #     dir_road_cube6_out_c, period_short, dir_road_cube6, dir_bg_boundaries, dir_income_bg
    # )
    # vis.scatter_income_vs_congestion(
    #     metrics_df_list_d, metrics_df_list_c, mode='disrupted_net',
    # )
    # vis.scatter_income_vs_congestion(
    #     metrics_df_list_d, metrics_df_list_c, mode='disrupted_net', expand=True
    # )
    # vis.scatter_income_vs_congestion(
    #     metrics_df_list_d, metrics_df_list_c, mode='complete_net', expand=True
    # )
    # vis.scatter_income_vs_congestion(
    #     metrics_df_list_d, metrics_df_list_c, mode='diff', expand=True
    # )

    # # congestion vs inundation
    # import pandas as pd
    # congestion_df_container = []
    # for d in [dir_road_cube6_out_d, dir_road_cube6_out_c]:
    #     congestion_df_list = pp_r.get_congestion_metrics(
    #         d, period_short, dir_road_cube6, dir_bg_boundaries, dir_income_bg
    #     )
    #     congestion_df = pd.concat([d[['congestion']] for d in congestion_df_list], axis=1)
    #     congestion_df['income'] = congestion_df_list[0]['income']
    #     congestion_df.columns = period_short + ['income']
    #     congestion_df_container.append(congestion_df)
    # road_segment = pp_r.import_road_seg_w_inundation_info(dir_road_inundated, speed_assigned)
    # road_segment = pp_r.add_geo_unit(road_segment, dir_bg_boundaries, ['COUNTYFP', 'TRACTCE', 'BLKGRPCE'])
    # road_segment = pp_r.merge_roads_demographic_bg(
    #     road_segment, pp_i.import_demographic(dir_income_bg, ['B19013_001E'],  ['9901'])
    # )
    # road_segment = road_segment[road_segment['bridge'].isna()]
    # depth_cols = [i for i in road_segment.columns if i.startswith('max_depth')]
    # road_segment[depth_cols] = (road_segment[depth_cols] * 30.48).clip(upper=road_cutoff_threshold)
    # severity_metrics_list = road_segment.groupby('id').apply(
    #     lambda group: pp_r.calculate_severity_metric_by_period(
    #         group, road_cutoff_threshold, 'cm', depth_cols, period_split, period_dict
    #     )
    # )
    # severity_df = pd.DataFrame(
    #     [[i[0], i[1], i[2], i[3], j] for i, j in severity_metrics_list.tolist()],
    #     columns=period_short + ['income'], index=severity_metrics_list.index
    # )
    # bg_geo = gpd.read_file(dir_bg_boundaries)
    # bg_geo['id'] = bg_geo['COUNTYFP'] + bg_geo['TRACTCE'] + bg_geo['BLKGRPCE']
    # vis.scatter_inundation_severity_vs_congestion(severity_df, congestion_df_container, bg_geo, expand=True)
    # vis.map_inundation_severity_and_congestion(severity_df, congestion_df_container, bg_geo)
    #####

    ###### Other vis
    # incidents = calculate_incidents_with_gis_travel_time(op=1)
    # _, _, _, geo_units, _ = calculate_incidents_metrics(incidents, 'income')
    # # vis.map_error(geo_units, dir_bg_boundaries, 'diff_travel', exclude_idx=[5293, 698],)
    # vis.map_geo_only(
    #     geo_units, dir_bg_boundaries, 'diff_travel', 'Estimation<br>error (s)', 'error',
    #     exclude_idx=[5293, 698], color_scale='RdBu', color_range=[-1400, 1400],
    #     geo_range=[-76.227827, 36.712386, -75.933628, 36.931997], legend_top=0.92
    # )

    # _, _, demo, _, _ = calculate_incidents_metrics(
    #     calculate_incidents_with_gis_travel_time(op=1),
    #     'age'
    # )
    # vis.map_demographic_geo_only(
    #     demo, dir_bg_boundaries,
    #     # 'demographic_value', 'Median house income<br>(US dollar)', '_demographic_income',
    #     # 'demographic_value', 'Percent Bachelor\'s<br>degree and higher', '_demographic_edu',
    #     # 'demographic_value', 'Percent non-minority', '_demographic_minority',
    #     'demographic_value', 'Percent population<br>aged 5-65', '_demographic_age',
    #     exclude_idx=[5293, 698],
    # )

    # vis.line_cut_n_ave_wellness(
    #     geo_units_normal,
    #     [round(x * 0.05, 2) for x in range(2, 11)],
    #     'range',
    # )

    ###### Other data processing: edit the net in CUBE
    # save_inundated_roads_4_sim()

    # nets = pp_r.merge_inundation_info_2_net(
    #     dir_road_cube7, dir_road_cube6_inundated, period_dict, period_split,
    #     'cubedb__Master_Network_CUBE7__link'
    # )
    # updated_nets = pp_r.edit_net_using_inundation(nets)
    # pp_r.edit_net_sqlite(
    #     dir_road_cube7, dir_road_cube7_inundated, updated_nets, period_split,
    #     'cubedb__Master_Network_CUBE7__link',
    # )

    ########

    print('End of the program.')
