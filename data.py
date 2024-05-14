import utils.preprocess as pp
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xyzservices.providers as xyz
from collections import Counter
import plotly.figure_factory as ff


if __name__ == "__main__":

    # preprocess and save ambulance data
    data = pp.rawData('./data/ambulance/virginiaBeach_ambulance_timeData.csv')
    data = pp.addOrigin(data, './data/rescueTeamLocation/rescueStations.txt')
    data = pp.geoCoding(data.loc['2013-01-01' : '2013-12-31', :], './data/ambulance/geocoded_saved/20130101-20131231.csv')

    roads = pp.readRoads('./data/roads/Streets.shp')
    roads = pp.makeSurface4Lines(roads, './data/roads/Road_Surfaces.shp', scale = 2.7)
    roads = pp.getWaterDepthOnRoads(roads, './data/inundation/tifData/depth_objID_35.tif', './data/inundation/croppedByRoads/croppedByRoads.tif')

    roads.drop(['line', 'midpoint','buffers','buffersUnscaled'], axis = 1).to_file('./data/roads/savedInundatedRoads/roads_with_objID_35.shp')
    roads['line'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_line.shp')
    roads['midpoint'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_midpoint.shp')
    roads['buffers'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_buffers.shp')
    roads['buffersUnscaled'].to_file('./data/roads/savedInundatedRoads/roads_with_objID_35_buffersUnscaled.shp')

    # # consider bridges in road network (PENDING)
    # bridges = gpd.read_file('./data/roads/bridges/bridgePolygon.shp').to_crs(str(inundation.crs))
    # inundation_cropped = inundationCutter(inundation, bridges, True, True)


