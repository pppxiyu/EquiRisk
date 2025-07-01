import utils.preprocess_incidents as pp_i
from config_vb import *


def calculate_incidents_with_gis_travel_time(op):
    """
    Modeling results have been finished in ArcGIS. The results are read here for the following analysis.
    :param op: option of the travel time estimation setting
    :return: Dataframe of incident
    """

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
        # OP2: real origins of ambulances are used
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
        # OP3: daily congestion is considered
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
        # OP4: flood day congestion is considered
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
        # OP5: both flood day congestion and real origins are considered
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
    """
    After reading the incidents information, calculate metrics in this function
    :param inc: Dataframe of incidents
    :param demo_label: demographic factors considers
    :return:
         icd_f: incident information during flood
         icd_n: incident information during normal time
         demo: demographic information
         g_units_f: incident information per geographic units (e.g., block group) during flood
         g_units_n: incident information per geographic units (e.g., block group) during normal time
    """

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


