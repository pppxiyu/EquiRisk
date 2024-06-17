import utils.preprocess_incidents as pp_i
import utils.preprocess_graph as pp_g
import utils.preprocess_roads as pp_r
import utils.preprocess as pp
import utils.modeling as mo
import data as dd
import warnings
import arcpy
import geopandas as gpd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    crs_geo = 'epsg:4326'
    crs_prj = 'epsg:32633'

    # road_segment = gpd.read_file(f"./data/roads/road_segment_vb.geojson")
    # turn_restriction = pp_r.import_turn_restriction('./data/roads/turn_restriction_vb_overpass.geojson')
    # road_segment = pp_r.add_roads_max_speed(
    #     road_segment,
    #     {
    #         'motorway': 55, 'motorway_link': 55, 'trunk': 55, 'trunk_link': 55,
    #         'primary': 55, 'primary_link': 55, 'secondary': 55, 'secondary_link': 55,
    #         'tertiary': 25, 'tertiary_link': 25, 'unclassified': 25, 'residential': 25, 'service': 25,
    #     }
    # )
    # road_segment = pp_r.add_travel_time_2_seg(road_segment)

    rescue_station = pp.import_rescue_station('./data/rescue_team_location/rescue_stations_n_nearest_geo.csv')
    incidents = pp_i.import_incident('./data/incidents/geocoded/20160101-20161015.csv')
    incidents = pp_i.add_actual_rescue_station(incidents, rescue_station)
    incidents = pp_i.add_nearest_rescue_station(incidents, rescue_station)
    incidents = pp_i.add_period_label(
        incidents, {
            '2016-10-09 00:00:00': 25,
            '2016-10-09 23:00:00': 48,
        }
    )

    geodatabase_addr = './gis_analysis/arcgis_emergency_service_routing/arcgis_emergency_service_routing.gdb'
    fd_name = 'road_network'
    nd_name = 'road_nd'
    nd_layer_name = 'road_nd_layer'




    # assign records to graph edge
    incidents = pp.assignGraphEdge(incidents, roads, 'RescueSquadPoint', 'OriginRoadID', 'Origin2RoadDist')
    incidents = pp.assignGraphEdge(incidents, roads, 'IncidentPoint', 'DestinationID', 'Destination2RoadDist')

    # find nearest rescue station
    # data = nearestRescueStation(data, rescue)

    # find the top nearest rescue stations
    incidents = pp.nearnessObediance(incidents, rescue, graph)

    # calculate the shortest path length and ave speed
    incidents = pp.assumedAveSpeed(incidents, rescue, graph)

    # calculate ratios
    graph = mo.runRoutingWithDisruption(graph, rescue, roads)
    graph = mo.getDisruptionRatio(graph)
    graph = mo.removeDisconnectedNodes(graph)
