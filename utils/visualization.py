import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


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


def scatter_demo_vs_error_w_period(
        df, col_demo, col_error, col_color, col_time, xaxis, yaxis,
        reg_line=None,
):
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
        df, xaxis, yaxis='Travel time<br>estimation error (s)',
        reg_line=None, color='#3F6F8C', size=17.5,
        col_demo='demographic_value', col_error='diff_travel',
        save_label=None, op_label='',
):
    xrange = extend_and_round_range([df[col_demo].min(), df[col_demo].max()], extension_percent=0.05)
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
    fig.update_layout(width=570, height=475)
    fig.show(renderer="browser")
    if save_label is not None:
        if reg_line is not None:
            p_label = 'f'
        else:
            p_label = 'n'
        fig.write_image(
            f"./manuscripts/figs/scatter_demo_vs_error_{save_label}_{p_label}{op_label}.png", engine="orca",
            width=570, height=475, scale=3.125
        )


def get_geo_bg(dir_bg, target_county_num='810'):
    import geopandas as gpd
    geo = gpd.read_file(dir_bg)
    geo = geo[geo['COUNTYFP'] == target_county_num]
    return geo


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


def map_geo_only(
        demo, dir_geo, color_col,
        demo_name, save_label,
        target_county_num='810', exclude_idx=None,
        color_scale='Blues_r', color_range=None,
        geo_range=None, legend_top=0.97,
):
    import json
    polygons = get_geo_bg(dir_geo, target_county_num=target_county_num)
    if exclude_idx is not None:
        polygons = polygons.drop(exclude_idx)
    if geo_range is None:
        minx, miny, maxx, maxy = polygons.total_bounds
    else:
        minx, miny, maxx, maxy = geo_range[0], geo_range[1], geo_range[2], geo_range[3],
    polygons_update = polygons[['geometry']].sjoin(demo, how='left', predicate='within')
    assert polygons_update['geometry'].duplicated().any() == False
    polygons_missing = polygons_update[polygons_update[color_col].isna()].copy()

    fig = px.choropleth(
        polygons_update,
        geojson=polygons_update.geometry, locations=polygons_update.index,
        color=color_col, color_continuous_scale=color_scale, range_color=color_range,
        projection="mercator",
    )
    fig.add_trace(
        go.Choropleth(
            geojson=json.loads(polygons_missing.geometry.to_json()),
            locations=polygons_missing.index,
            z=[1] * len(polygons_missing),
            colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
            showscale=False
        )
    )
    fig.update_traces(
        marker_line_color="#383838",
        marker_line_width=0.75,
    )
    fig.update_geos(
        center=dict(lat=(miny + maxy) / 2, lon=(minx + maxx) / 2),
        lataxis=dict(range=[miny - 0.01, maxy + 0.01]), lonaxis=dict(range=[minx - 0.01, maxx + 0.01]),
        visible=False
    )
    fig.update_coloraxes(
        colorbar=dict(
            len=0.5,
            title=dict(
                text=demo_name,
                font=dict(size=18)
            ),
            outlinecolor='black',
            outlinewidth=0.5,
            tickfont=dict(size=14),
            x=0.9, xanchor="left",
            y=legend_top, yanchor="top"
        ),
    )
    fig.update_layout(
        font=dict(family="Arial", color="Black"),
        autosize=False,
        width=600,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/map_demographic_{save_label}.png", engine="orca",
        width=600, height=600, scale=3.125
    )



def map_demographic(gdf, color_col, demo_name):
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


def map_road_speed(gdf, time_col, label=''):
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
        color_discrete_map=interval_color_map,
        mapbox_style='light',
    )
    fig.update_layout(
        legend=dict(
            title="Speed intervals (mph)",
            x=0.03,
            y=0.01,
            xanchor="left",
            yanchor="bottom",
        ),
        mapbox=dict(
            center=dict(
                lat=36.773,
                lon=-76.068,
            ),
            zoom=10.15
        ),
        width=610,
        height=800,
    )
    # fig.for_each_trace(lambda trace: trace.update(visible='legendonly') if trace.name == 'Zero speed' else ())
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/map_traffic{label}.png", engine="orca",
        width=610, height=800, scale=3.125
    )


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
    if mode == 'actual':
        fig.add_trace(  # actual match
            go.Scattermapbox(
                lat=matches_y.flatten(),
                lon=matches_x.flatten(),
                mode='lines',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='grey',
                ),
                showlegend=True,
                name='Recorded dispatches'
            )
        )
    elif mode == 'nearest':
        matches_x_c, matches_y_c = get_matches(gdf_i_closed)
        fig.add_trace(
            go.Scattermapbox(
                lat=matches_y_c.flatten(),
                lon=matches_x_c.flatten(),
                mode='lines',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='#F2B680',
                ),
                showlegend=True,
                name='Nearest dispatches but<br>station closed'
            )
        )
        matches_x_o, matches_y_o = get_matches(gdf_i_occupied)
        fig.add_trace(
            go.Scattermapbox(
                lat=matches_y_o.flatten(),
                lon=matches_x_o.flatten(),
                mode='lines',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='#AA4A44',
                ),
                showlegend=True,
                name='Nearest dispatches but<br>station occupied'
            )
        )
    fig.add_trace(
        go.Scattermapbox(  # station big circle
            lat=gdf_station['lat'],
            lon=gdf_station['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=15,
                color='#191970',
            ),
            showlegend=False,
        ),
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
            showlegend=True,
            name='Emergency service stations',
        )
    )
    fig.add_trace(  # incidents bg
        go.Scattermapbox(
            lat=gdf_incidents.geometry.y,
            lon=gdf_incidents.geometry.x,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=12,
                color='#191970',
            ),
            showlegend=False
        )
    )
    fig.add_trace(  # incidents
        go.Scattermapbox(
            lat=gdf_incidents.geometry.y,
            lon=gdf_incidents.geometry.x,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color='#75238C',
            ),
            showlegend=True,
            name='Emergency service incidents'
        )
    )
    fig.update_layout(
        mapbox=dict(
            accesstoken=open("./utils/mapboxToken.txt").read(),
            style="light",
            zoom=11,
            center=dict(lat=36.840, lon=-76.095),
        ),
        width=700,
        height=650,
        showlegend=True,
        paper_bgcolor="black",
        plot_bgcolor="white",
        margin=dict(l=2, r=2, t=2, b=2),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(80,80,80,1)",
            borderwidth=1,
            font=dict(
                size=26, family="Arial", color="black",
            )
        ),
        # fig.update_layout(
        #     mapbox_layers=[  # nearest match
        #         {
        #             "sourcetype": "geojson",
        #             "type": "line",
        #             "color": "#F2B680",
        #             "line": {"dash": [2.5, 1]},
        #             "source": gj_c,
        #         },
        #         {
        #             "sourcetype": "geojson",
        #             "type": "line",
        #             "color": "#AA4A44",
        #             "line": {"dash": [2.5, 1]},
        #             "source": gj_o,
        #         },
        #     ],
        # )
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/map_origin_shift_{mode}.png", engine="orca",
        width=700, height=650, scale=3.125
    )

    return


def scatter_dist_icd_travel_time(icd, period, mode='dist'):
    import warnings
    import pandas as pd
    import numpy as np
    flood_begins = list(period.keys())[0]
    flood_ends = list(period.keys())[1]

    # clean
    icd['TravelTime'] = icd['TravelTime'].dt.total_seconds()
    icd['TravelTime'] = icd['TravelTime'].replace(0, np.nan)
    icd_mean = icd['TravelTime'].mean()
    icd_std = icd['TravelTime'].std()
    icd['TravelTime'] = icd['TravelTime'].where(
        (icd['TravelTime'] >= icd_mean - icd_std * 3) & (icd['TravelTime'] <= icd_mean + icd_std * 3), np.nan
    )

    # conditioned on incident locations
    icd_flood = icd[(icd['Call Date and Time'] >= flood_begins) & (icd['Call Date and Time'] <= flood_ends)]
    icd_normal = icd[(icd['Call Date and Time'] < flood_begins) | (icd['Call Date and Time'] > flood_ends)]
    icd_flood = icd_flood.to_crs('epsg:26918')
    icd_normal = icd_normal.to_crs('epsg:26918')
    icd_normal_ave = icd_normal[['TravelTime', 'IncidentAddress', 'geometry']].dissolve(
        'IncidentAddress', as_index=False, aggfunc='mean',
    )
    icd_flood = icd_flood.sjoin_nearest(icd_normal_ave, how='left', distance_col='sjoin_dist')
    icd_normal = icd_normal.sjoin_nearest(icd_normal_ave, how='left', distance_col='sjoin_dist')
    if icd_flood['sjoin_dist'].max() > 0:
        warnings.warn(f"Max dist to the nearest incident loc is {icd_flood['sjoin_dist'].max()}")
    icd_flood['travel_icrs_ratio'] = icd_flood['TravelTime_left'] / icd_flood['TravelTime_right']
    icd_normal['travel_icrs_ratio'] = icd_normal['TravelTime_left'] / icd_normal['TravelTime_right']

    # vis
    if mode == 'dist':
        import plotly.figure_factory as ff
        icd_flood_d = icd_flood[~icd_flood['travel_icrs_ratio'].isna()]
        icd_normal_d = icd_normal[~icd_normal['travel_icrs_ratio'].isna()].sample(1000)
        icd_normal_d = icd_normal_d[icd_normal_d['travel_icrs_ratio'] <= 10]
        fig = ff.create_distplot(
            [icd_flood_d['travel_icrs_ratio'].to_list(), icd_normal_d['travel_icrs_ratio'].to_list()],
            ['Flooding', 'Non-flooding'],
            bin_size=.5, show_rug=False, colors=['#235689', '#42BCB2']
        )
        fig.update_layout(
            xaxis=dict(
                title='Travel time / Baseline time',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat=',',
                zeroline=False,
                nticks=10,
                range=[0, 10],
                tickvals=[0, 1, 2, 4, 6, 8, 10],
            ),
            yaxis=dict(
                title='Probability',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat=',',
                zeroline=False,
                range=[0, 1],
            ),
            font=dict(family="Arial", size=18, color="black"),
            width=450, height=450,
            legend=dict(x=0.75, y=0.95, xanchor="center", yanchor="top",),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.add_shape(
            type='line',
            x0=1, x1=1, y0=0, y1=1,
            line=dict(color='black', width=1, dash='dash')
        )
        fig.show(renderer="browser")
        fig.write_image(
            "./manuscripts/figs/dist_travel_increase.png", engine="orca",
            width=450, height=450, scale=3.125
        )
    elif mode == 'scatter':
        icd_flood['label'] = ['Flooding'] * len(icd_flood)
        icd_normal['label'] = ['Non-flooding'] * len(icd_normal)
        icd_flood_s = icd_flood[~icd_flood['travel_icrs_ratio'].isna()]
        icd_normal_s = icd_normal[~icd_normal['travel_icrs_ratio'].isna()].sample(100)
        icd_updated = pd.concat([icd_normal_s, icd_flood_s,])
        fig = px.scatter(
            icd_updated, x="TravelTime_right", y="travel_icrs_ratio", color="label", symbol="label",
            color_discrete_map={'Non-flooding': '#42BCB2', 'Flooding': '#235689', }
        )
        fig.update_layout(
            xaxis=dict(
                title='Baseline time',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat=',',
                zeroline=False,
                nticks=10,
                range=[0, 900],
            ),
            yaxis=dict(
                title='Travel time / Baseline time',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat=',',
                zeroline=False,
                range=[0, 10],
                tickvals=[0, 1, 2, 4, 6, 8, 10]
            ),
            font=dict(family="Arial", size=18, color="black"),
            width=450, height=450,
            legend=dict(x=0.8, y=0.95, xanchor="center", yanchor="top", title_text=None),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.add_shape(
            type='line',
            x0=0, x1=900, y0=1, y1=1,
            line=dict(color='black', width=1, dash='dash')
        )
        fig.show(renderer="browser")
        fig.write_image(
            "./manuscripts/figs/scatter_travel_increase.png", engine="orca",
            width=450, height=450, scale=3.125
        )
    return


def extend_and_round_range(original_range, extension_percent=0.1,):
    import math
    min_val, max_val = original_range
    current_length = max_val - min_val

    extension = current_length * extension_percent
    new_min = min_val - extension
    new_max = max_val + extension

    if min_val == 0:
        scale = 10 ** math.floor(math.log10(max_val))
    else:
        scale = 10 ** math.floor(math.log10(min_val))

    new_min_rounded = math.floor(new_min / scale) * scale
    new_max_rounded = math.ceil(new_max / scale) * scale

    return [new_min_rounded, new_max_rounded]


def bar_wellness(lower, higher):
    lower_r = round(lower, 3)
    higher_r = round(higher, 3)
    x = ['Lower-income<br>class', 'Middle- and<br>upper-<br>income class']
    y = [lower_r, higher_r]
    fig = go.Figure(data=[go.Bar(
        x=x, y=y,
        text=y,
        textposition='auto',
        textfont=dict(size=16, color='white'),
        marker_color=['#62B197', '#E18E6D'],
    )])
    fig = layout(fig)
    fig.update_layout(
        xaxis=dict(
            showline=True,
            linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',',
            zeroline=False,
            nticks=10,
        ),
        yaxis=dict(
            title='Risk assessment bias',
            showline=True,
            linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',',
            range=[-0.8, 0],
            zeroline=False,
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=400, height=400,
        font=dict(size=16)
    )
    fig.show(renderer="browser")
    fig.write_image(
        "./manuscripts/figs/bar_bias.png", engine="orca",
        width=400, height=400, scale=3.125
    )


def bar_capacity_short(gdf):
    pass
    return

def scatter_income_service_volumn(incidents, closing_info):
    import pandas as pd
    station_ave_income = incidents[['Rescue Squad Number', 'demographic_value']].groupby(
        'Rescue Squad Number'
    ).mean()
    station_volume = incidents[['Rescue Squad Number', 'demographic_value']].groupby(
        'Rescue Squad Number'
        ).count().rename(columns={'demographic_value': 'count'})
    stations = pd.concat([station_ave_income, station_volume], axis=1)
    stations['status'] = ['Operating'] * len(stations)
    stations = stations.reset_index()

    problem_station = closing_info[
        (closing_info['if_nearest_occupied'] == True) | closing_info['if_nearest_closed'] == True
        ]['Number_nearest'].unique()
    stations.loc[stations['Rescue Squad Number'].isin(problem_station), 'status'] = 'Disrupted'

    fig = px.scatter(
        stations, x='demographic_value', y='count', color='status',
        color_discrete_map={'Operating': '#808080', 'Disrupted': '#75238C',},
        trendline="ols", trendline_scope="overall"
    )
    fig.update_layout(
        xaxis=dict(
            title='Average household income<br>of incidents served (USD)',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        yaxis=dict(
            title='Incidents served count',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        font=dict(family="Arial", size=18, color="black"),
        width=465, height=450,
        legend=dict(
            x=0.85, y=0.95, xanchor="center", yanchor="top", title_text=None,
            bordercolor='#808080', borderwidth=1.5,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.update_traces(
        selector=dict(mode='lines'), showlegend=False,
        line=dict(color='#808080', dash='dash', width=1)
    )
    fig.update_traces(
        selector=dict(mode='markers'),
        marker=dict(size=10)
    )
    fig.show(renderer="browser")
    fig.write_image(
        "./manuscripts/figs/scatter_income_volume.png", engine="orca",
        width=465, height=450, scale=3.125
    )
    return


def scatter_inundation_severity_vs_income(df):
    fig = px.scatter(
        df, x='income', y='severity', trendline="ols", trendline_scope="overall"
    )
    fig.update_layout(
        xaxis=dict(
            title='Median household income (USD)<br>',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        yaxis=dict(
            title='Road inundation severity',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            x=0.85, y=0.95, xanchor="center", yanchor="top", title_text=None,
            bordercolor='#808080', borderwidth=1.5,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=465, height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.update_traces(
        selector=dict(mode='lines'), showlegend=False,
        line=dict(color='#712773', dash='dash', width=2)
    )
    fig.update_traces(
        selector=dict(mode='markers'),
        marker=dict(size=6, color='#777AA6')
    )
    fig.show(renderer="browser")
    fig.write_image(
        "./manuscripts/figs/scatter_income_severity.png", engine="orca",
        width=465, height=450, scale=3.125
    )
    return


def scatter_income_vs_congestion(df_list):
    import pandas as pd
    df = pd.concat(df_list, axis=0, ignore_index=True)

    color_list = ['#0FA66E', '#8A9A5B', '#D98A29', '#5FB6D9']
    fig = px.scatter(
        df, x='income', y='congestion', color='period',
        color_discrete_sequence=color_list
    )
    fig.update_layout(
        xaxis=dict(
            title='Median household income (USD)',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        yaxis=dict(
            title='Congestion severity',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            x=0.75, y=1, xanchor="center", yanchor="top", title_text=None,
            bordercolor='#808080', borderwidth=1.5, font=dict(size=16)
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=465, height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/scatter_income_congestion.png", engine="orca",
        width=465, height=450, scale=3.125
    )
    return