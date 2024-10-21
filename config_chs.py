from config_vb import *

city_name = [(33.00, 32.67, -79.79, -80.20), 'chs']

geodatabase_addr = './gis_analysis/emerg_routing_CHS/emerg_routing_CHS.gdb'

period_dict = {
    '...': None,
}  # NOTE: UTC TIME !!!

speed_assigned = {
    'motorway': 55, 'motorway_link': 55, 'trunk': 55, 'trunk_link': 55,
    'primary': 55, 'primary_link': 55, 'secondary': 55, 'secondary_link': 55,
    'tertiary': 25, 'tertiary_link': 25, 'unclassified': 25, 'residential': 25, 'service': 25,
}

dir_rescue_station = 'data/CHS/rescue_team_location/rescue_stations.txt'
dir_rescue_station_n_nearest_geo = 'data/CHS/rescue_team_location/rescue_stations_n_nearest_geo.csv'
dir_turn = 'data/CHS/roads/turn_restriction_CHS_overpass.geojson'
dir_road_folder = './data/CHS/roads'
dir_road = './data/CHS/roads/road_segment_chs.geojson'

dir_road_inundated = 'data/CHS/roads/road_segment_CHS.geojson'
dir_inaccessible_routes = "..."

dir_incidents = '...'
dir_incidents_routing_nearest = '...'

# null params
period_split = None
period_short = None
dir_tract_boundaries = ''
dir_bg_boundaries = ''
dir_income_tract = ''
dir_income_bg = ''
dir_edu = ''
dir_edu_bg = ''
dir_population = ''
dir_minority_bg = ''
dir_age_bg = ''
dir_road_cube6 = ''
dir_road_cube6_out_c = ''
dir_road_cube6_out_d = ''
dir_match_osm_n_VDOT = ''
dir_road_cube6_inundated = ""
dir_road_cube7 = ''
dir_road_cube7_inundated = ''
