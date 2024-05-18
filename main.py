import utils.preprocess as pp
import utils.modeling as mo


if __name__ == "__main__":

    # reload data and build geoDataFrame
    incidents = pp.reload_incidents('./data/ambulance/geocoded_saved/20160101-20161015.csv')
    roads = pp.reload_roads('./data/roads/savedInundatedRoads/roads_with_objID_35.shp')

    # create graph
    graph = pp.roads2Graph(roads)
    # showGraphRoads(roads, graph)
    # NOTE: the graph is un-directed right now, the logic should be checked if changed to directed

    # read the location of rescue squads and attach them to nodes
    rescue = pp.readRescue('./data/rescue_team_location/rescueStations.txt', 'EPSG:4326', roads)

    # additional info for descriptive analysis
    # assign records to graph edges
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
