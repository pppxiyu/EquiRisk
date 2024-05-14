import utils.preprocess as pp
import utils.modeling as mo


if __name__ == "__main__":

    # reload data and build geoDataFrame
    data = pp.reLoadData('./data/ambulance/geocoded_saved/20160101-20161015.csv')
    roads = pp.reLoadRoads('./data/roads/savedInundatedRoads/roads_with_objID_35.shp')

    # create graph
    graph = pp.roads2Graph(roads)
    # showGraphRoads(roads, graph)
    # NOTE: the graph is un-directed right now, the logic should be checked if changed to directed

    # read the location of rescue squads and attach them to nodes
    rescue = pp.readRescue('./data/rescueTeamLocation/rescueStations.txt', 'EPSG:4326', roads)

    # additional info for descriptive analysis
    # assign records to graph edges
    data = pp.assignGraphEdge(data, roads, 'RescueSquadPoint', 'OriginRoadID', 'Origin2RoadDist')
    data = pp.assignGraphEdge(data, roads, 'IncidentPoint', 'DestinationID', 'Destination2RoadDist')

    # find nearest rescue station
    # data = nearestRescueStation(data, rescue)

    # find the top nearest rescue stations
    data = pp.nearnessObediance(data, rescue, graph)

    # calculate the shortest path length and ave speed
    data = pp.assumedAveSpeed(data, rescue, graph)

    # calculate ratios
    graph = mo.runRoutingWithDisruption(graph, rescue, roads)
    graph = mo.getDisruptionRatio(graph)
    graph = mo.removeDisconnectedNodes(graph)
