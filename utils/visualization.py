import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def s_mapbox_scatter_px(gdf, color_col):
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.scatter_mapbox(
        gdf,
        lat=gdf.geometry.y,
        lon=gdf.geometry.x,
        color=color_col,
        color_continuous_scale='Blues_r',
        zoom=10,
        size=len(gdf) * [3],
    )
    fig.show(renderer="browser")


def s_mapbox_choropleth_px_continuous(gdf, color_col):
    import json
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.choropleth_mapbox(
        gdf,
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        color=color_col,
        color_continuous_scale='Blues',
        zoom=10,
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        opacity=0.75,
    )
    fig.show(renderer="browser")


def s_mapbox_choropleth_px_divergent(gdf, color_col,):
    import json
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.choropleth_mapbox(
        gdf,
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        color=color_col,
        color_continuous_scale='RdBu',
        zoom=10,
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        opacity=0.75,
        range_color=[-gdf[color_col].abs().max(), gdf[color_col].abs().max()],
    )
    fig.show(renderer="browser")


def s_mapbox_choropleth_scatter(demo_gdf, demo_col, incident_gdf, incident_col):
    import json
    demo_gdf = demo_gdf.to_crs('epsg:32633')
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.choropleth_mapbox(
        demo_gdf,
        geojson=json.loads(demo_gdf.to_json()),
        locations=demo_gdf.index,
        color=demo_col,
        color_continuous_scale="Blues",
        zoom=10,
        center={"lat": demo_gdf.geometry.centroid.y.mean(), "lon": demo_gdf.geometry.centroid.x.mean()},
        opacity=0.75,
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=incident_gdf.geometry.y,
            lon=incident_gdf.geometry.x,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=15,
                color=incident_gdf[incident_col],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(x=0.9, xanchor="left", title="Bias",),
            ),
            text=incident_gdf[incident_col],
        )
    )
    fig.update_layout(coloraxis_colorbar=dict(x=0.8, xanchor="left", title="Demographic",))
    fig.show(renderer="browser")


def layout(fig):
    fig.update_layout(
        font=dict(
            family="Arial",
            size=18,
            color="Black"
        ),
        autosize=False,
        width=800,
        height=400,
    )
    return fig


def scatter_demo_vs_error_w_period(df, col_demo, col_error, col_color, col_time, xaxis, yaxis, reg_line=None):
    df[col_color] = df[col_color].astype(float)
    color_map = {row[col_color]: row[col_time] for index, row in df.iterrows()}
    fig = px.scatter(
        df, x=col_demo, y=col_error,
        color=col_color, color_continuous_scale='Emrld'
    )

    if reg_line is not None:
        slope = reg_line[0]
        intercept = reg_line[1]
        x_line = df[col_demo]
        y_line = x_line * slope + intercept
        fig = fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(dash='dash'),
                showlegend=False
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=xaxis,
            showline=True,
            linewidth=1.5,
            linecolor='black',
            showgrid=False,
            ticks='inside'
        ),
        yaxis=dict(
            title=yaxis,
            showline=True,
            linewidth=1.5,
            linecolor='black',
            showgrid=False,
            ticks='inside',
            range=[-2500, 1000],
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_coloraxes(
        colorbar_title=dict(
            text='Average incident time',
            font=dict(size=18)  # Set color legend title font size
        ),
        colorbar=dict(
            outlinecolor='black',
            outlinewidth=1.5,
            tickvals=list(color_map.keys()),
            ticktext=list(color_map.values()),
            tickfont=dict(size=12)
        ),
    )
    fig.update_traces(
        marker=dict(
            size=25,
            line=dict(color='black', width=1),
            opacity=0.75,
        )
    )
    fig = layout(fig)
    fig.show(renderer="browser")


def scatter_demo_vs_error(
        df, col_demo, col_error, xaxis, yaxis, xrange,
        reg_line=None, color='#3F6F8C', size=17.5, height=None,
):
    fig = px.scatter(
        df, x=col_demo, y=col_error,
    )

    fig.add_hline(y=0, line_width=1)

    if reg_line is not None:
        slope = reg_line[0]
        intercept = reg_line[1]
        x_line = df[col_demo]
        y_line = x_line * slope + intercept
        fig = fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(dash='5px, 5px'),
                showlegend=False
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=xaxis,
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            range=xrange,
            zeroline=False,
            nticks=10,
        ),
        yaxis=dict(
            title=yaxis,
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            range=[-2000, 1000],
            zeroline=False,
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_traces(
        marker=dict(
            size=size,
            line=dict(color='black', width=1),
            opacity=0.75,
            color=color,
        )
    )
    fig = layout(fig)
    if height is not None:
        fig.update_layout(height=height)
    fig.show(renderer="browser")


def map_error(gdf, color_col):
    import json
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.choropleth_mapbox(
        gdf,
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        color=color_col,
        color_continuous_scale='RdBu',
        zoom=10,
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        opacity=0.75,
        range_color=[-1500, 1500],
    )
    fig.update_coloraxes(
        colorbar_title=dict(
            text='Travel time estimation error (s)',
            font=dict(size=18)
        ),
        colorbar=dict(
            outlinecolor='black',
            outlinewidth=1,
            tickfont=dict(size=12)
        ),
    )
    fig = layout(fig)
    fig.show(renderer="browser")


def map_demo(gdf, color_col, demo_name):
    import json
    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    fig = px.choropleth_mapbox(
        gdf,
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        color=color_col,
        color_continuous_scale='Blues_r',
        zoom=10,
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        opacity=0.75,
    )
    fig.update_coloraxes(
        colorbar_title=dict(
            text=demo_name,
            font=dict(size=18)
        ),
        colorbar=dict(
            outlinecolor='black',
            outlinewidth=1,
            tickfont=dict(size=12)
        ),
    )
    fig = layout(fig)
    fig.show(renderer="browser")


def line_cut_n_ave_wellness(geo_units, threshold_percent, cut_mode,):

    lower_mean = []
    upper_mean = []
    lower_median = []
    upper_median = []
    threshold_list = []
    for p in threshold_percent:

        if cut_mode == 'range':
            threshold = (
                    p
                    * (geo_units['demographic_value'].max() - geo_units['demographic_value'].min())
                    + geo_units['demographic_value'].min()
            )
        elif cut_mode == 'num':
            threshold = geo_units['demographic_value'].quantile(p)

        threshold_list.append(threshold)
        lower_mean.append(geo_units[geo_units['demographic_value'] <= threshold]['wellness'].mean())
        upper_mean.append(geo_units[geo_units['demographic_value'] > threshold]['wellness'].mean())
        lower_median.append(geo_units[geo_units['demographic_value'] <= threshold]['wellness'].median())
        upper_median.append(geo_units[geo_units['demographic_value'] > threshold]['wellness'].median())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=threshold_list, y=lower_mean, name='Lower income - average wellness',
                             line=dict(color='#30A9D9', width=3)))
    fig.add_trace(go.Scatter(x=threshold_list, y=lower_median, name='Lower income - median wellness',
                             line=dict(color='#0477BF', width=3)))
    fig.add_trace(go.Scatter(x=threshold_list, y=upper_mean, name='Higher income - average wellness',
                             line=dict(color='#F2A25C', width=3)))
    fig.add_trace(go.Scatter(x=threshold_list, y=upper_median, name='Higher income - median wellness',
                             line=dict(color='#F5BF53', width=3)))

    fig = layout(fig)
    fig.update_layout(
        xaxis=dict(
            title='Threshold (US dollar)',
            showline=True,
            linewidth=1.5,
            linecolor='black',
            showgrid=False,
            ticks='inside',
            nticks=5,
            ticktext=[f'{v} ({p * 100:.0f}%)' for v, p in zip(threshold_list, threshold_percent)],
            tickvals=threshold_list
        ),
        yaxis=dict(
            title='Wellness',
            showline=True,
            linewidth=1.5,
            linecolor='black',
            showgrid=False,
            ticks='inside',
            range=[-1, 1],
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            x=0.4,
            y=0.99,
            traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0.6)',
        ),
        width=700,
        height=600,
    )
    fig.show(renderer="browser")
    return


def map_road_speed(gdf, time_col):
    import shapely
    import numpy as np
    import pandas as pd

    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    gdf = gdf.to_crs('epsg:4326')

    speed_column = gdf['DISTANCE_x'] / gdf[time_col] * 60
    speed_column[speed_column <= 1e-1] = 0
    speed_column[speed_column >= 65] = speed_column[(speed_column < 65) & (speed_column > 1e-1)].median()

    num_colors = 13
    null_name = 'Zero speed'
    color_scale = px.colors.sample_colorscale(
        px.colors.diverging.RdYlGn, [i/(num_colors-1) for i in range(num_colors)]
    )
    gdf['interval'] = pd.cut(
        speed_column,
        precision=2,
        bins=np.linspace(0, 65, num_colors + 1),
    )
    gdf['interval_name'] = gdf['interval'].apply(lambda x: f"{x.left}-{x.right}")
    interval_color_map = {interval: color for interval, color in zip(
        gdf['interval_name'].unique().categories.tolist() + [null_name],
        color_scale + ['rgb(220,220,220)']
    )}
    gdf = gdf.sort_values(by='interval')
    gdf['interval_name'] = gdf['interval_name'].astype(str)
    gdf['interval_name'] = gdf['interval_name'].replace('nan', null_name)

    lats = []
    lons = []
    colors = []
    for lines, color in zip(gdf.geometry, gdf['interval_name'].tolist()):
        if isinstance(lines, shapely.geometry.linestring.LineString):
            linestrings = [lines]
        elif isinstance(lines, shapely.geometry.multilinestring.MultiLineString):
            linestrings = lines.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            colors = np.append(colors, [color] * len(x))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            colors = np.append(colors, [color])
    fig = px.line_mapbox(
        lat=lats,
        lon=lons,
        color=colors,
        zoom=10.7,
        color_discrete_map=interval_color_map,
        mapbox_style='light',
    )
    fig.update_layout(
        legend=dict(
            title="Speed intervals (mile/h)",
        )
    )
    fig.for_each_trace(lambda trace: trace.update(visible='legendonly') if trace.name == 'Zero speed' else ())
    fig.show(renderer="browser")


def map_origin_shift(gdf_incidents, gdf_station, mode='nearest'):
    import numpy as np
    import geojson

    def get_matches(gdf):
        matches = gdf[['geometry_x', 'geometry_y']].values
        matches_x = np.column_stack((
            np.array([[p.x for p in a] for a in matches]),
            np.full((matches.shape[0], 1), None)
        ))
        matches_y = np.column_stack((
            np.array([[p.y for p in a] for a in matches]),
            np.full((matches.shape[0], 1), None)
        ))
        return matches_x, matches_y

    def build_geojson(gdf):
        gdf = gdf[['geometry_x', 'geometry_y']]
        features = []

        for _, row in gdf.iterrows():
            line_coords = [
                (row['geometry_x'].x, row['geometry_x'].y),
                (row['geometry_y'].x, row['geometry_y'].y)
            ]
            feature = geojson.Feature(
                geometry=geojson.LineString(line_coords)
            )
            features.append(feature)

        feature_collection = geojson.FeatureCollection(features)
        return feature_collection

    px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
    gdf_incidents = gdf_incidents.to_crs('epsg:4326')
    gdf_station = gdf_station.to_crs('epsg:4326')

    gdf_i_1 = gdf_incidents.merge(gdf_station, how='left', left_on='Number_nearest', right_on='Number')
    gdf_i_closed = gdf_i_1[gdf_i_1['if_nearest_closed'] == True]
    gdf_i_occupied = gdf_i_1[gdf_i_1['if_nearest_occupied'] == True]
    gj_c = build_geojson(gdf_i_closed)
    gj_o = build_geojson(gdf_i_occupied)

    gdf_i_2 = gdf_incidents.merge(gdf_station, how='left', left_on='Number_actual', right_on='Number')
    matches_x, matches_y = get_matches(gdf_i_2)

    fig = go.Figure()
    fig.add_trace(  # station big circle
        go.Scattermapbox(
            lat=gdf_station['lat'],
            lon=gdf_station['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=15,
                color='#191970',
            ),
            showlegend=False,
        )
    )
    fig.add_trace(  # station small circle
        go.Scattermapbox(
            lat=gdf_station['lat'],
            lon=gdf_station['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='#6F8FAF',
            ),
        )
    )
    fig.add_trace(  # incidents
        go.Scattermapbox(
            lat=gdf_incidents.geometry.y,
            lon=gdf_incidents.geometry.x,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='#0F52BA',
            ),
        )
    )
    if mode == 'actual':
        fig.add_trace(  # actual match
            go.Scattermapbox(
                lat=matches_y.flatten(),
                lon=matches_x.flatten(),
                mode='lines',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='#89CFF0',
                ),
            )
        )
    elif mode == 'nearest':
        fig.update_layout(
            mapbox_layers=[  # nearest match
                {
                    "sourcetype": "geojson",
                    "type": "line",
                    "color": "grey",
                    "line": {"dash": [2.5, 1]},
                    "source": gj_c,
                },
                {
                    "sourcetype": "geojson",
                    "type": "line",
                    "color": "#AA4A44",
                    "line": {"dash": [2.5, 1]},
                    "source": gj_o,
                },
            ],
        )
    fig.update_layout(
        mapbox=dict(
            accesstoken=open("./utils/mapboxToken.txt").read(),
            style="carto-positron",
            zoom=11,
            center=dict(lat=36.835, lon=-76.08),
        ),
        # mapbox_layers=[  # nearest match
        #     {
        #         "sourcetype": "geojson",
        #         "type": "line",
        #         "color": "grey",
        #         "line": {"dash": [2.5, 1]},
        #         "source": gj_c,
        #     },
        #     {
        #         "sourcetype": "geojson",
        #         "type": "line",
        #         "color": "#AA4A44",
        #         "line": {"dash": [2.5, 1]},
        #         "source": gj_o,
        #     },
        # ],
        width=1000,
        height=800,
        showlegend=False,
    )
    fig.show(renderer="browser")

    return


def reg_spatial_lag(df, x, y, k):
    from pysal.model import spreg
    from pysal.lib import weights

    knn = weights.KNN.from_dataframe(df, k=k)
    reg = spreg.GM_Lag(
        df[[y]].values,
        df[[x]].values,
        name_y=y,
        name_x=[x],
        w=knn,
    )
    print(reg.summary)
    return reg


def reg_regular(df, x, y):
    from pysal.model import spreg

    reg = spreg.OLS(
        df[[y]].values,
        df[[x]].values,
        name_y=y,
        name_x=[x],
    )
    print(reg.summary)


def reg_z_score_4_compared_coeff(a1, a2, std1, std2, cov):
    import math
    return (a1 - a2) / ( math.sqrt(std1 ** 2 + std2 ** 2 - 2 * cov) )


def reg_t_score_4_compared_coeff(a1, a2, std1, std2, cov):
    import math
    return (a1 - a2) / ( math.sqrt(std1 ** 2 + std2 ** 2 - 2 * cov) )


def reg_SUR(x1, y1, x2, y2,):
    from collections import OrderedDict
    import statsmodels.api as sm
    from linearmodels.system import SUR
    import pandas as pd
    pd.options.display.float_format = '{:.8f}'.format
    equations = OrderedDict()
    equations['model1'] = {
        'dependent': y1,
        'exog': sm.add_constant(x1),
    }
    equations['model2'] = {
        'dependent': y2,
        'exog': sm.add_constant(x2),
    }
    mod = SUR(equations)
    res = mod.fit(
        method='ols',
        full_cov=True,
        iterate=False,
        iter_limit=500,
        tol=1e-6,
        cov_type='unadjusted'
    )
    print(res.cov)
    return res.cov.loc['model1_exog_0.1', 'model2_exog_1.1']



