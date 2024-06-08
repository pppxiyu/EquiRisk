import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import OrderedDict

# import rasterio
# from rasterio.plot import show
# import rasterio.mask
import networkx as nx
from shapely.geometry import Polygon, Point
import warnings

# import libpysal
# import contextily as cx


def import_rescue_station(addr, crs='epsg:4326'):
    rescue = pd.read_csv(addr)
    rescue = gpd.GeoDataFrame(
        rescue,
        geometry=gpd.points_from_xy(rescue['lon'], rescue['lat'])).set_crs(crs)
    return rescue


def add_nearest_segment(point, road_segment):
    i = point.distance(road_segment.geometry).sort_values().index[0]
    id = road_segment.iloc[i]['osmid']
    return id


def add_nearest_intersection(point, road_intersection):
    i = point.distance(road_intersection.geometry).sort_values().index[0]
    id = road_intersection.iloc[i]['osmid']
    return id


# visualzation
def showRoadsInundation(inundation):
    fig = plt.figure(figsize=(100, 50))
    ax = fig.add_subplot()
    ax = show(inundation, ax=ax, cmap='pink')
    roads.plot(ax=ax)
    plt.show()


def showInundation(inundation):
    plt.imshow(inundation.read()[0], cmap='hot')
    plt.colorbar()
    plt.show()


def showBridgesInundation():
    fig = plt.figure(figsize=(100, 50))
    ax = fig.add_subplot()
    ax = show(inundation, ax=ax, cmap='pink')
    bridges.plot(ax=ax)
    plt.show()


def showLinesSurfaces_withBounds(roads, roadSurfaces, bounds):
    fig = plt.figure(figsize=(100, 50))
    ax = fig.add_subplot()
    roadsBounded = roads.cx[bounds[0]: bounds[1], bounds[2]: bounds[3]]
    roadSurfacesBounded = roadSurfaces.cx[bounds[0]: bounds[1], bounds[2]: bounds[3]]
    roadsBounded.plot(ax=ax, color='red')
    roadSurfacesBounded.plot(ax=ax)
    plt.show()


def showMidpointLineSurface(roads, roadSurfaces):
    fig = plt.figure(figsize=(400, 200))
    ax = fig.add_subplot()
    roads.line.plot(ax=ax, linewidth=.75, zorder=0)
    # roads.midpoint.plot(ax = ax, zorder = 0)
    roadSurfaces.geometry.plot(ax=ax, color='red', zorder=0)
    plt.show()


def getGlobalBounds(gpd):
    # get global bounds of a geopandas df
    xmin = gpd.bounds.minx.min()
    xmax = gpd.bounds.maxx.max()
    ymin = gpd.bounds.miny.min()
    ymax = gpd.bounds.maxy.max()
    return xmin, xmax, ymin, ymax


def getMiddleBounding(bounds, percent=0.05):
    xmin = bounds[0]
    xmax = bounds[1]
    ymin = bounds[2]
    ymax = bounds[3]
    xminNew = xmin + ((xmax - xmin) * ((1 - percent) / 2))
    xmaxNew = xminNew + (xmax - xmin) * percent
    yminNew = ymin + ((ymax - ymin) * ((1 - percent) / 2))
    ymaxNew = yminNew + (ymax - ymin) * percent
    return xminNew, xmaxNew, yminNew, ymaxNew





# additional info

def _nearestRescue4Incidents(data, rescue):
    # find nearest rescues for all incidents
    incidents = data.DestinationID.values
    voronoi = nx.voronoi_cells(graph, set(rescue.OBJECTID_nearestRoad.unique()), weight='weight')
    nearestRescue = []
    for incident in incidents:
        len1 = len(nearestRescue)
        if np.isnan(incident):
            nearestRescue.append(np.nan)
        else:
            for key, value in voronoi.items():
                if int(incident) in list(value):
                    if key == 'unreachable':
                        nearestRescue.append(np.nan)
                    else:
                        nearestRescue.append(key)
                    break
        len2 = len(nearestRescue)
        if len2 == len1:
            print(incident, 'not in any')
    return nearestRescue


def _generateDistDf(rescue, graph):
    nodeList = range(1, len(list(graph.nodes())) + 1)
    df = pd.DataFrame(nodeList, index=nodeList, columns=['NodeNames'])
    for res in rescue.values:
        resName = res[0]
        resRoadNumber = res[-1]
        distanceDict = nx.single_source_dijkstra_path_length(graph, resRoadNumber, weight='weight')
        orderedResRoadNumber = OrderedDict(sorted(distanceDict.items()))
        orderedResRoadNumberDf = pd.DataFrame.from_dict(orderedResRoadNumber, orient='index',
                                                        columns=['from' + resName])
        orderedResRoadNumberDf = orderedResRoadNumberDf.reset_index()
        df = df.merge(orderedResRoadNumberDf, how='left', left_on='NodeNames', right_on='index').drop(columns='index')
    return df


def _obedianceOfShortestPrinciple(Series, distanceDataFrame):
    DestinationID = Series.DestinationID
    RescueSquadNumber = Series.RescueSquadNumber
    if len(distanceDataFrame[distanceDataFrame.NodeNames == DestinationID]) != 0:
        # NOTE:some incidents are not considered because no road around them
        allDist = list(np.sort(distanceDataFrame[distanceDataFrame.NodeNames == DestinationID].values[0][1:]))
        realDist = distanceDataFrame[distanceDataFrame.NodeNames == DestinationID]['from' + RescueSquadNumber].values[0]
        if np.isnan(realDist):
            # NOTE: some roads are disconnected even in normal time
            realDistRank = np.nan
            realDistIncreaseRatio = np.nan
        else:
            realDistRank = allDist.index(realDist) + 1
            if allDist[0] == 0:
                # NOTE: in case the incident is just beside the rescue station, set the dist to 1
                allDist[0] = 1
            realDistIncreaseRatio = realDist / allDist[0]
    else:
        realDistRank = np.nan
        realDistIncreaseRatio = np.nan
    return realDistRank, realDistIncreaseRatio


def _shortestRouteLength_slowLegacy(row, graph, ifPrintError=False):
    try:
        length = nx.dijkstra_path_length(graph, row.OriginRoadID, row.DestinationID, weight='weight')
    except BaseException as ex:
        if ifPrintError == True:
            print(ex)
        length = np.nan
    return length


def _shortestRouteLength(s, distanceDataFrame):
    if np.isnan(s.DestinationID):
        return np.nan
    else:
        return distanceDataFrame[distanceDataFrame.NodeNames == s.DestinationID]['from' + s.RescueSquadNumber].values[0]


def nearestRescueStation(data, rescue):
    # find nearest rescue station
    data['NearestRescue'] = _nearestRescue4Incidents(data, rescue)
    data = data.merge(rescue.loc[:, ["OBJECTID_nearestRoad", "Number"]], how='left', left_on="NearestRescue",
                      right_on='OBJECTID_nearestRoad')
    data = data.drop(columns='OBJECTID_nearestRoad').rename(columns={"Number": "NearestRescueNumber"})
    return data


def nearnessObediance(data, rescue, graph):
    # find the top nearest rescue stations
    distanceDataFrame = _generateDistDf(rescue, graph)
    obediance = data.apply(_obedianceOfShortestPrinciple, distanceDataFrame=distanceDataFrame, axis=1,
                           result_type='expand')
    data['NearestOrder'] = obediance[0]
    data['DisobediancePathIncrease'] = obediance[1]
    return data


def assumedAveSpeed(data, rescue, graph):
    # calculate shortest path length and ave speed
    distanceDataFrame = _generateDistDf(rescue, graph)
    data['AssumedRouteLength'] = data.apply(_shortestRouteLength, distanceDataFrame=distanceDataFrame, axis=1)
    data['AverageSpeed'] = data['AssumedRouteLength'] / data['TravelTime']
    return data
