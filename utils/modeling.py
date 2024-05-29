import networkx as nx
import math
import numpy as np
import arcpy


def init_service_area_analysis_arcgis(nd_layer_name, rescue_station, cutoff_time: list):

    note = """
        MANUAL OPERATION: go to the network dataset under the feature dataset. Right click the nd, and 
        add a new travel mode called 'DriveTime' manually. This step must be finished in the UI, as the 
        travel mode attribute of nd is set as read-only. Esri does not support adding a travel mode from 
        scrtach to nd yet. Ref: https://community.esri.com/t5/arcgis-network-analyst-questions/
        how-to-add-travel-mode-when-creating-a-network/td-p/1042568.
    """

    try:
        service_area = arcpy.nax.ServiceArea(nd_layer_name)
    except ValueError as e:
        if str(e) == "Input network dataset does not have at least one travel mode.":
            raise ValueError(note)
        else:
            raise e

    service_area.timeUnits = arcpy.nax.TimeUnits.Seconds
    service_area.defaultImpedanceCutoffs = cutoff_time
    service_area.travelMode = arcpy.nax.GetTravelModes(nd_layer_name)['DriveTime']
    service_area.outputType = arcpy.nax.ServiceAreaOutputType.Polygons
    service_area.geometryAtOverlap = arcpy.nax.ServiceAreaOverlapGeometry.Split

    fields = ["Name", "SHAPE@"]
    with service_area.insertCursor(arcpy.nax.ServiceAreaInputDataType.Facilities, fields) as cur:
        for i, row in rescue_station.iterrows():
            name, lat, lon = row['Number'], row['lat'], row['lon']
            cur.insertRow(([name, (lon, lat)]))

    return service_area


def init_route_analysis_arcgis(nd_layer_name, rescue_station, incidents, num_sampling=None):

    note = """
        MANUAL OPERATION: go to the network dataset under the feature dataset. Right click the nd, and 
        add a new travel mode called 'DriveTime' manually. This step must be finished in the UI, as the 
        travel mode attribute of nd is set as read-only. Esri does not support adding a travel mode from 
        scrtach to nd yet. Ref: https://community.esri.com/t5/arcgis-network-analyst-questions/
        how-to-add-travel-mode-when-creating-a-network/td-p/1042568.
    """

    try:
        route = arcpy.nax.Route(nd_layer_name)
    except ValueError as e:
        if str(e) == "Input network dataset does not have at least one travel mode.":
            raise ValueError(note)
        else:
            raise e

    route.timeUnits = arcpy.nax.TimeUnits.Seconds
    route.travelMode = arcpy.nax.GetTravelModes(nd_layer_name)['DriveTime']


    if num_sampling is not None:
        incidents = incidents.sample(n=num_sampling)
    rescue_station_involved = incidents['Rescue Squad Number'].unique()
    rescue_station = rescue_station[rescue_station['Number'].isin(rescue_station_involved)]

    fields = ["RouteName", "Sequence", "SHAPE@"]
    with route.insertCursor(arcpy.nax.RouteInputDataType.Stops, fields) as cur:
        for res in rescue_station.iterrows():
            for inc in incidents[incidents['Rescue Squad Number'] == res[1]["Number"]].iterrows():
                route_name = f'{res[1]["Number"]}-{inc[1]["EMS Call Number"]}'
                cur.insertRow(([
                    route_name, 1,
                    (res[1]['lon'], res[1]['lat'])]))
                cur.insertRow(([route_name, 2, (inc[1]['lon'], inc[1]['lat'])]))

    return route





def _addPathLen2Graph(graph, rescue, weight, newAttribute_rescueSquad, newAttribute_pathLen,
                      newAttribute_pathList=None):
    # some roads are disconnected from all the rescue station even in normal time (as the raw data indicates)
    voronoi = nx.voronoi_cells(graph, set(rescue.OBJECTID_nearestRoad.unique()), weight=weight)
    for rescueSquad, destinations in zip(voronoi.keys(), voronoi.values()):
        if rescueSquad == 'unreachable':
            print(len(destinations), 'nodes are unreachable when building voronoi for', newAttribute_pathLen)
            for des in destinations:
                graph.nodes[des][newAttribute_rescueSquad] = np.nan
                graph.nodes[des][
                    newAttribute_pathLen] = math.inf  # set path len to inf if it's disconnected from rescues
        #                 print('NOTE: node', des, 'is unreachable when building voronoi for', newAttribute_pathLen)
        else:
            for des in destinations:
                shortestPathLen = nx.shortest_path_length(graph, source=rescueSquad, target=des, weight=weight)
                graph.nodes[des][newAttribute_pathLen] = shortestPathLen
                graph.nodes[des][newAttribute_rescueSquad] = rescueSquad
                if newAttribute_pathList:
                    shortestPath = nx.shortest_path(graph, source=rescueSquad, target=des, weight=weight)
                    graph.nodes[des][newAttribute_pathList] = shortestPath
                if shortestPathLen == 0:
                    graph.nodes[des][newAttribute_pathLen] = 1
                if shortestPathLen == math.inf:
                    graph.nodes[des][newAttribute_rescueSquad] = math.inf
    return graph, voronoi


def _addDisruption(graph, roads, newAttribute='weightWithDisruption', threshold=3):
    nx.set_edge_attributes(graph, nx.get_edge_attributes(graph, "weight"), newAttribute)
    disruptedRoads = roads[roads['waterDepth'] >= threshold]['OBJECTID'].to_list()
    for disruption in disruptedRoads:
        for edge in graph.edges(disruption):
            graph.edges()[edge][newAttribute] = math.inf  # set edge weight to inf if it's disrupted by inundation
    return graph


def _changeValue4DisruptedRoad(roads, graph, threshold=3):
    # the disrupted road itself is not disconnected, so assign the shortestPath of adjancent road to this road
    for disruption in roads[roads['waterDepth'] >= threshold]['OBJECTID'].to_list():
        pathLen = []
        edgeNum = []
        for edge in graph.edges(disruption):
            pathLen.append(graph.nodes()[edge[1]]['shortestPathLenWithDisruption'])
            edgeNum.append(edge[1])
        if pathLen != []:  # in case there are disconnected single node
            graph.nodes()[disruption]['shortestPathLenWithDisruption'] = min(pathLen)
            if min(pathLen) != math.inf:
                graph.nodes()[disruption]['rescueAssignedWithDisruption'] = edgeNum[pathLen.index(min(pathLen))]
            else:
                graph.nodes()[disruption]['rescueAssignedWithDisruption'] = np.nan
    return graph


def runRoutingWithDisruption(graph, rescue, roads):
    graph, _ = _addPathLen2Graph(graph, rescue, 'weight', 'rescueAssigned', 'shortestPathLen', 'shortestPathList')
    graphDisrupted = _addDisruption(graph, roads, threshold=1)
    graph, _ = _addPathLen2Graph(graphDisrupted, rescue, 'weightWithDisruption', 'rescueAssignedWithDisruption',
                                 'shortestPathLenWithDisruption', 'shortestPathListWithDisruption')
    graph = _changeValue4DisruptedRoad(roads, graph, threshold=1)
    return graph


def getDisruptionRatio(graph):
    nx.set_node_attributes(graph,
                           {x[0]: y[1] / x[1] if y[1] / x[1] != math.inf else np.nan \
                            for x, y in zip(nx.get_node_attributes(graph, "shortestPathLen").items(),
                                            nx.get_node_attributes(graph,
                                                                   "shortestPathLenWithDisruption").items())},
                           'travelTimeIncreaseRatio')
    roads['travelTimeIncreaseRatio'] = roads['OBJECTID'].map(
        nx.get_node_attributes(graph, "travelTimeIncreaseRatio"))
    return graph


def removeDisconnectedNodes(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = [node for node in graph.nodes if node not in largest_cc]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

