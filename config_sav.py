from config_vb import *

geodatabase_addr = './gis_analysis/arcgis_emergency_service_routing/arcgis_emergency_service_routing.gdb'

period_dict = {
    '2016-10-09 00:00:00': 25,
    '2016-10-09 23:00:00': 48,
}  # NOTE: UTC TIME !!!
period_split = {
    'AM_Peak': [9, 13],
    'Midday_Off-peak': [13, 19],
    'PM_Peak': [19, 22],
    'Midnight_Off-peak': [22, 9],
}  # NOTE: UTC TIME from EDT !!!  2016-10-09 was in EDT.  Local time  + 4 hours = UTC time.
period_short = ['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP']
speed_assigned = {
    'motorway': 55, 'motorway_link': 55, 'trunk': 55, 'trunk_link': 55,
    'primary': 55, 'primary_link': 55, 'secondary': 55, 'secondary_link': 55,
    'tertiary': 25, 'tertiary_link': 25, 'unclassified': 25, 'residential': 25, 'service': 25,
}

dir_rescue_station_n_nearest_geo = 'data/VB/rescue_team_location/rescue_stations_n_nearest_geo.csv'
dir_incidents = 'data/VB/incidents/geocoded/20160101-20161015.csv'
dir_incidents_routing_nearest = 'data/VB/incidents/incidents_w_routing_nearest/incidents_w_routing_nearest.csv'
dir_turn = 'data/VB/roads/turn_restriction_vb_overpass.geojson'
dir_road = "data/VB/roads/road_segment_vb.geojson"
dir_road_inundated = "data/VB/roads/road_segment_vb_inundated.geojson"
dir_inaccessible_routes = "data/VB/incidents/inaccessible_route"
dir_tract_boundaries = 'data/VB/boundaries/cb_2016_51_tract_500k/cb_2016_51_tract_500k.shp'
dir_bg_boundaries = 'data/VB/boundaries/tl_2017_51_bg/tl_2017_51_bg.shp'
dir_income_tract = 'data/VB/demographic/S1901/ACSST5Y2016.S1901-Data.csv'
dir_income_bg = 'data/VB/demographic/B19013/ACSDT5Y2016.B19013-Data.csv'
dir_edu = 'data/VB/demographic/S1501/ACSST5Y2016.S1501-Data.csv'
dir_edu_bg = 'data/VB/demographic/B15003/ACSDT5Y2016.B15003-Data_adapted.csv'
dir_population = 'data/VB/demographic/DP05/ACSDP5Y2016.DP05-Data.csv'
dir_minority_bg = 'data/VB/demographic/B02001/ACSDT5Y2016.B02001-Data_adapted.csv'
dir_age_bg = 'data/VB/demographic/B01001/ACSDT5Y2016.B01001-Data_adapted.csv'
dir_road_cube6 = 'data/VB/HR_Model_V2_04302024/trueshp/HR_Model_trueshp08022022.shp'
dir_road_cube6_out_c = 'data/VB/HR_Model_V2_04302024/dbf_output/complete_net'
dir_road_cube6_out_d = 'data/VB/HR_Model_V2_04302024/dbf_output/disrupted_net'
dir_match_osm_n_VDOT = 'data/VB/roads/osm_match_VDOT.geojson'
dir_road_cube6_inundated = "data/VB/roads/road_segment_4_sim_vb_inundated.geojson"
dir_road_cube7 = 'data/VB/HR_Model_V2_04302024/network_converted/network_CUBE7.SQLite'
dir_road_cube7_inundated = 'data/VB/HR_Model_V2_04302024/network_converted_inundated'
