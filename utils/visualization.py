import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def mapbox_scatter_px(gdf, color_col):
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.scatter_mapbox(
        gdf,
        lat=gdf.geometry.y,
        lon=gdf.geometry.x,
        color=color_col,
        color_continuous_scale='ice',
        zoom=10,
        size=len(gdf) * [3],
    )
    fig.show(renderer="browser")
