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

    speed_column = gdf['DISTANCE_shp'] / gdf[time_col] * 60
    speed_column[speed_column <= 1e-1] = 0
    speed_column[speed_column >= 65] = speed_column[(speed_column < 65) & (speed_column > 1e-1)].median()

    num_colors = 13
    null_name = 'Zero or cutoff'
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


def merge_income_service_volumn_status(incidents, closing_info):
    import pandas as pd
    station_ave_income = incidents[['Rescue Squad Number', 'demographic_value']].groupby(
        'Rescue Squad Number'
    ).mean()
    station_volume = incidents[['Rescue Squad Number', 'demographic_value']].groupby(
        'Rescue Squad Number'
        ).count().rename(columns={'demographic_value': 'count'})
    stations = pd.concat([station_ave_income, station_volume], axis=1)
    stations['status'] = ['Available'] * len(stations)
    stations = stations.reset_index()

    problem_station = closing_info[
        (closing_info['if_nearest_occupied'] == True) | closing_info['if_nearest_closed'] == True
        ]['Number_nearest'].unique()
    stations.loc[stations['Rescue Squad Number'].isin(problem_station), 'status'] = 'Unavailable'
    return stations


def scatter_income_service_volumn(incidents, closing_info, plot='scatter'):
    stations = merge_income_service_volumn_status(incidents, closing_info)

    if plot == 'scatter':
        fig_scatter = px.scatter(
            stations, x='demographic_value', y='count', color='status',
            color_discrete_map={'Available': '#808080', 'Unavailable': '#75238C',},
            trendline="ols", trendline_scope="overall"
        )
        fig_scatter.update_layout(
            xaxis=dict(
                title='Average household income of area served (USD)',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat=',',
                zeroline=False,
            ),
            yaxis=dict(
                title='Service count',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat=',',
                zeroline=False,
            ),
            font=dict(family="Arial", size=18, color="black"),
            width=850 * 0.65, height=225,
            legend=dict(
                x=0.9, y=1, xanchor="center", yanchor="top", title_text=None,
                bordercolor='#808080', borderwidth=1.5,
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig_scatter.update_traces(
            selector=dict(mode='lines'), showlegend=False,
            line=dict(color='#808080', dash='dash', width=1)
        )
        fig_scatter.update_traces(
            selector=dict(mode='markers'),
            marker=dict(size=10)
        )
        fig_scatter.show(renderer="browser")
        fig_scatter.write_image(
            "./manuscripts/figs/scatter_income_volume.png", engine="orca",
            width=850 * 0.65, height=225,
            scale=3.125
        )

    elif plot == 'dist':
        import plotly.figure_factory as ff
        fig_dist = ff.create_distplot(
            [
                stations[stations['status'] == 'Available']['demographic_value'].tolist(),
                stations[stations['status'] == 'Unavailable']['demographic_value'].tolist()
            ],
            ['Available', 'Unavailable'],
            show_hist=False, show_rug=False,
            colors=['#808080', '#75238C']
        )
        fig_dist.update_layout(
            xaxis=dict(
                title=' ',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickformat='~s',
                zeroline=False,
            ),
            yaxis=dict(
                title='Probability',
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                zeroline=False,
                exponentformat="e",
            ),
            font=dict(family="Arial", size=18, color="black"),
            legend=dict(
                x=0.8, y=1.1, xanchor="center", yanchor="top", title_text=None,
                font=dict(size=16),
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            width=850 * 0.35, height=225,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig_dist.show(renderer="browser")
        fig_dist.write_image(
            "./manuscripts/figs/dist_station_income.png", engine="orca",
            width=850 * 0.35, height=225,
            scale=3.125
        )

    else:
        raise ValueError('plot could be "scatter" or "dist"')

    return


def scatter_inundation_severity_vs_income(df):
    fig = px.scatter(
        df, x='income', y='severity', trendline="ols", trendline_scope="overall"
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
            title='Road inundation<br>severity',
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
        # width=465, height=450,
        width=850, height=275,
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
        width=850, height=275, scale=3.125
    )
    return


def scatter_income_vs_congestion(df_list_d, df_list_c, mode='disrupted_net', expand=False):
    import pandas as pd
    if mode == 'disrupted_net':
        df = pd.concat(df_list_d, axis=0, ignore_index=True)
    elif mode == 'complete_net':
        df = pd.concat(df_list_c, axis=0, ignore_index=True)
    elif mode == 'diff':
        df_d = pd.concat(df_list_d, axis=0, ignore_index=True)
        df_c = pd.concat(df_list_c, axis=0, ignore_index=True)
        df = df_d.copy()
        df['congestion'] = df_d['congestion'] - df_c['congestion']
    df['congestion'] = df['congestion'] * 100

    color_list = ['#992F87', '#C9A2C6', '#A5CFE3', '#55759E',]
    fig = px.scatter(
        df, x='income', y='congestion', color='period',
        color_discrete_sequence=color_list,
        # trendline='ols',
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
            title='Travel time<br>increase (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
            range=[-10, 210]
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            x=0.5, y=1.2, xanchor="center", yanchor="middle", title_text=None,
            font=dict(size=18), orientation="h", itemwidth=30,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=600, height=250,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    if mode == 'diff':
        fig.update_layout(yaxis=dict(range=[-100, 200]), width=610, height=250)
        fig.show(renderer="browser")
        fig.write_image(
            f"./manuscripts/figs/scatter_income_congestion_{mode}.png", engine="orca",
            width=610, height=250, scale=3.125
        )
    else:
        fig.show(renderer="browser")
        fig.write_image(
            f"./manuscripts/figs/scatter_income_congestion_{mode}.png", engine="orca",
            width=600, height=250, scale=3.125
        )

    if expand:
        for n in [t.name for t in fig.data]:
            for trace in fig.data:
                if trace.name not in [n]:
                    trace.visible = False
                else:
                    trace.visible = True
            if mode == 'diff':
                fig.update_layout(yaxis=dict(range=[-100, 200]), width=610,)
                fig.show(renderer="browser")
                fig.write_image(
                    f"./manuscripts/figs/scatter_income_congestion_{mode}_{n}.png", engine="orca",
                    width=610, scale=3.125
                )
            else:
                fig.show(renderer="browser")
                fig.write_image(
                    f"./manuscripts/figs/scatter_income_congestion_{mode}_{n}.png", engine="orca",
                    width=600, height=250, scale=3.125
                )
    return


def bar_per_non_nearest(per_1, per_2):
    fig = go.Figure(data=[
        go.Bar(
            name='Not nearest', x=['Non-flooding', 'Flooding'], y=[per_1, per_2],
            marker=dict(color='#42BCB2'),
            text=[f'{per_1 * 100:.2f}%', f'{per_2 * 100:.2f}%'], textposition='outside',
            textfont=dict(color='white')
        ),
        go.Bar(
            name='Nearest', x=['Non-flooding', 'Flooding'], y=[1 - per_1, 1 - per_2],
            marker=dict(color='#235689'),
            text=[f'{(1 - per_1) * 100:.2f}%', f'{(1 - per_2) * 100:.2f}%'], textposition='inside',
            insidetextanchor='start',
            textfont=dict(color='white')
        )
    ])
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        yaxis=dict(
            tickvals=[0, 1], ticktext=['0%', '100%'],
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(font=dict(size=16), orientation="h", yanchor="bottom", xanchor="center", y=1.05, x=0.5,),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=450, height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/bar_percent_nearest.png", engine="orca",
        width=450, height=450, scale=3.125
    )
    return

def bar_per_nearest_reason(icd_n, icd_f):
    icd_n = icd_n.dropna(subset=['if_nearest_occupied', 'if_nearest_closed'])
    icd_f = icd_f.dropna(subset=['if_nearest_occupied', 'if_nearest_closed'])
    per_n = [
        ((icd_n['if_nearest_occupied'] == False) & (icd_n['if_nearest_closed'] == False)).mean() * 100,
        icd_n['if_nearest_occupied'].mean() * 100,
        icd_n['if_nearest_closed'].mean() * 100
    ]
    assert sum(per_n) == 100
    per_f = [
        ((icd_f['if_nearest_occupied'] == False) & (icd_f['if_nearest_closed'] == False)).mean() * 100,
        icd_f['if_nearest_occupied'].mean() * 100,
        icd_f['if_nearest_closed'].mean() * 100
    ]
    assert sum(per_f) == 100

    fig = go.Figure(data=[
        go.Bar(
            name='Closure', x=['Non-flooding', 'Flooding'], y=[per_n[2], per_f[2]],
            marker=dict(color='#197AB7'),
            text=[f'{per_n[2]:.2f}%', f'{per_f[2]:.2f}%'], textposition='outside',
            textfont=dict(color='white')
        ),
        go.Bar(
            name='Occupation', x=['Non-flooding', 'Flooding'], y=[per_n[1], per_f[1]],
            marker=dict(color='#A5CFE3'),
            text=[f'{per_n[1]:.2f}%', f'{per_f[1]:.2f}%'], textposition='inside',
            insidetextanchor='start',
            textfont=dict(color='white')
        ),
        go.Bar(
            name='Others', x=['Non-flooding', 'Flooding'], y=[per_n[0], per_f[0]],
            marker=dict(color='#CFD4D7'),
            text=[f'{per_n[0]:.2f}%', f'{per_f[0]:.2f}%'], textposition='outside',
            textfont=dict(color='black'), cliponaxis=False
        )
    ])
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        yaxis=dict(
            tickvals=[0, 100], ticktext=['0%', '100%'],
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            font=dict(size=16), traceorder="normal",
            orientation="h", yanchor="bottom", xanchor="center", y=1.05, x=0.5,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=450, height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/bar_percent_nearest_reason.png", engine="orca",
        width=450, height=450, scale=3.125
    )
    return


def bar_ave_income_normal_disrupted_icd(
        income_n_nearest, income_n_not_nearest, income_f_nearest, income_f_not_nearest
):
    fig = go.Figure(data=[
        go.Bar(
            name='Nearest', x=['Non-flooding', 'Flooding'], y=[income_n_nearest, income_f_nearest],
            text=[f'{income_n_nearest:.2f}', f'{income_f_nearest:.2f}'], textposition='outside',
            marker=dict(color='#A5CFE3'),
        ),
        go.Bar(
            name='Not nearest', x=['Non-flooding', 'Flooding'], y=[income_n_not_nearest, income_f_not_nearest],
            text=[f'{income_n_not_nearest:.2f}', f'{income_f_not_nearest:.2f}'], textposition='outside',
            marker=dict(color='#235689'),
        )
    ])
    fig.update_layout(
        barmode='group',
        xaxis=dict(
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        yaxis=dict(
            title='Average household income<br>of incidents (USD)',
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            font=dict(size=16), traceorder="normal",
            orientation="h", yanchor="bottom", xanchor="center", y=1.05, x=0.5,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=450, height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/bar_income_normal_disrupted_icd.png", engine="orca",
        width=450, height=450, scale=3.125
    )
    return


def line_hotspot_ave_time(t_by_h, t_std_by_h, t_min_by_h, loc):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, )
    fig.add_trace(
        go.Scatter(
            x=list(range(24)), y=t_by_h, mode='lines+markers', name='Average travel time',
            line=dict(color='#235689',),
            marker=dict(color='#235689',)
        ), row=1, col=1,
    )
    # fig.add_trace(
    #     go.Scatter(x=list(range(24)), y=t_std_by_h, mode='lines+markers', name='Standard deviation'), row=2, col=1
    # )
    fig.add_shape(
        type="line", x0=0, x1=23, y0=min(t_by_h), y1=min(t_by_h),
        line=dict(color="grey", width=1, dash="dash"), xref="x1", yref="y1"
    )
    fig.add_annotation(
        x=23, y=min(t_by_h), text=f"{min(t_by_h):.1f}s", showarrow=False, xref="x1", yref="y1",
        xanchor="left", font=dict(color="grey", size=14)
    )
    fig.add_shape(
        type="line", x0=0, x1=23, y0=max(t_by_h), y1=max(t_by_h),
        line=dict(color="grey", width=1, dash="dash"), xref="x1", yref="y1"
    )
    fig.add_annotation(
        x=23, y=max(t_by_h), text=f"{max(t_by_h):.1f}s", showarrow=False, xref="x1", yref="y1",
        xanchor="left", font=dict(color="grey", size=14)
    )
    fig.update_layout(
        xaxis=dict(
            title='Hours in day',
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
            tickmode='array', tickvals=[0] + list(range(1, 24, 2)),
        ),
        yaxis=dict(
            title='Average travel<br>time (s)',
            showline=True, linewidth=2, linecolor='black', showgrid=False,
            ticks='outside', tickformat=',', zeroline=False,
        ),
        # xaxis2=dict(
        #     title='Hours in day',
        #     showline=True, linewidth=2, linecolor='black', showgrid=False,
        #     ticks='outside', tickformat=',', zeroline=False,
        # ),
        # yaxis2=dict(
        #     title='Seconds',
        #     showline=True, linewidth=2, linecolor='black', showgrid=False,
        #     ticks='outside', tickformat=',', zeroline=False,
        # ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            font=dict(size=16), traceorder="normal",
            orientation="h", yanchor="bottom", xanchor="center", y=1.05, x=0.5,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=600, height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/line_ave_travel_time_hour.png", engine="orca",
        width=600, height=400, scale=3.125
    )
    return


def scatter_inundation_severity_vs_congestion(
        inundation, congestion_container, block_group, spatial_lag=False, expand=False
):
    import pandas as pd
    congestion = congestion_container[0] - congestion_container[1]
    congestion['income'] = congestion_container[0]['income']
    congestion = congestion[~congestion.isna().any(axis=1)]
    inundation = inundation[~inundation.isna().any(axis=1)]

    congestion_melt = pd.melt(
        congestion.reset_index(), id_vars=['id', 'income'], value_vars=['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP'],
        var_name='period', value_name='congestion'
    )
    inundation_melt = pd.melt(
        inundation.reset_index(), id_vars=['id', 'income'], value_vars=['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP'],
        var_name='period', value_name='inundation'
    )
    combined = congestion_melt.merge(inundation_melt, on=['id', 'period'], how='inner')
    replacements = [('AM_PK', 'AM Peak'), ('Md_OP', 'Midday'), ('PM_PK', 'PM Peak'), ('Nt_OP', 'Night')]
    for old, new in replacements:
        combined['period'] = combined['period'].str.replace(old, new)
    x_name = 'inundation'

    if spatial_lag:
        from pysal.lib import weights
        import geopandas as gpd
        combined_w_bg = gpd.GeoDataFrame(combined[combined['period'] == combined['period'].unique()[0]].merge(
            block_group[['geometry', 'id']], on='id', how='left'
        ), geometry='geometry').dissolve(by='id').reset_index()
        m = weights.Queen.from_dataframe(combined_w_bg, use_index=False, silence_warnings=True)
        m.transform = 'r'
        adj_m = m.full()[0]
        lag_col = 'inundation'
        lag_n = 3
        for ln in range(lag_n):
            for p in combined['period'].unique():
                c = combined[combined['period'] == p][lag_col].values
                for _ in range(ln + 1):
                    c = adj_m @ c
                combined.loc[combined['period'] == p, f'{lag_col}_l{ln + 1}'] = c
        combined[f'{lag_col}_all_lag'] = combined[
            [lag_col] + [f'{lag_col}_l{ln + 1}' for ln in list(range(lag_n))]
        ].sum(axis=1)
        x_name = 'inundation_all_lag'

    # vis
    color_list = ['#992F87', '#C9A2C6', '#A5CFE3', '#55759E',]
    fig = px.scatter(
        combined,
        x=x_name,
        y='congestion',
        color='period',
        trendline="ols",
        color_discrete_sequence=color_list,
    )
    fig.update_layout(
        xaxis=dict(
            title='Road inundation severity',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
        ),
        yaxis=dict(
            title='Travel time<br>increase (%)',
            showline=True,
            linewidth=2,
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickformat=',',
            zeroline=False,
            range=[-1, 2]
        ),
        font=dict(family="Arial", size=18, color="black"),
        legend=dict(
            x=0.5, y=1.2, xanchor="center", yanchor="middle", title_text=None,
            font=dict(size=18), orientation="h", itemwidth=30,
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        width=600, height=250,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.visible = False
    fig.show(renderer="browser")
    fig.write_image(
        f"./manuscripts/figs/scatter_congestion_inundation.png", engine="orca",
        width=600, height=250, scale=3.125
    )
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.visible = True
    if expand:
        for n in [t.name for t in fig.data]:
            for trace in fig.data:
                if trace.name not in [n]:
                    trace.visible = False
                else:
                    trace.visible = True
            fig.show(renderer="browser")
            fig.write_image(
                f"./manuscripts/figs/scatter_congestion_inundation_{n}.png", engine="orca",
                width=600, height=250, scale=3.125
            )
    return


def map_inundation_severity_and_congestion(inundation, congestion_list, block_group):
    import geopandas as gpd
    block_group_vb = block_group[block_group['COUNTYFP'] == '810']
    block_group_vb = block_group_vb[~block_group_vb['id'].isin(['8109901000', '8100418024'])]
    inundation_w_geo = inundation.copy()
    inundation_w_geo = inundation_w_geo.reset_index().merge(block_group_vb[['geometry', 'id']], on='id')
    inundation_w_geo = gpd.GeoDataFrame(inundation_w_geo, geometry='geometry')

    congestion_d = congestion_list[0]
    congestion_w_geo_d = congestion_d.copy()
    congestion_w_geo_d = congestion_w_geo_d.reset_index().merge(block_group_vb[['geometry', 'id']], on='id')
    congestion_w_geo_d = gpd.GeoDataFrame(congestion_w_geo_d, geometry='geometry')
    assert (inundation_w_geo.columns == congestion_w_geo_d.columns).all(), 'Columns name inconsistent.'

    congestion_c = congestion_list[1]
    congestion_w_geo_c = congestion_c.copy()
    congestion_w_geo_c = congestion_w_geo_c.reset_index().merge(block_group_vb[['geometry', 'id']], on='id')
    congestion_w_geo_c = gpd.GeoDataFrame(congestion_w_geo_c, geometry='geometry')
    congestion_w_geo_change = congestion_w_geo_d.copy()
    congestion_w_geo_change[['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP']] = (
            congestion_w_geo_d[['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP']] -
            congestion_w_geo_c[['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP']]
    )

    def plot(gdf, value_col, value_name, save_label, color_scheme="Viridis", range_color=[0, 1]):
        import json
        px.set_mapbox_access_token(open("./utils/mapboxToken.txt").read())
        fig = px.choropleth(
            gdf,
            geojson=json.loads(gdf.to_json()),
            locations=gdf.index,
            color=value_col,
            color_continuous_scale=color_scheme,
            projection="mercator",
            range_color=range_color
        )
        fig.update_geos(
            center=dict(
                lat=(gdf.total_bounds[1] + gdf.total_bounds[3]) / 2,
                lon=(gdf.total_bounds[0] + gdf.total_bounds[2]) / 2
            ),
            lataxis=dict(range=[gdf.total_bounds[1] - 0.01, gdf.total_bounds[3] + 0.01]),
            lonaxis=dict(range=[gdf.total_bounds[0] - 0.01, gdf.total_bounds[2] + 0.01]),
            visible=False
        )
        fig.update_traces(
            marker_line_color="#36454F",
            marker_line_width=0.75,
        )
        fig.update_coloraxes(
            colorbar=dict(
                len=0.5,
                title=dict(text=value_name, font=dict(size=18)),
                outlinecolor='black',
                outlinewidth=0.5,
                tickfont=dict(size=14),
                x=0.82, xanchor="left",
                y=0.97, yanchor="top",
            ),
        )
        fig.update_layout(
            font=dict(family="Arial", color="Black"),
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.show(renderer="browser")
        fig.write_image(
            f"./manuscripts/figs/map_block_group_{save_label}_{value_col}.png", engine="orca",
            width=600, height=600, scale=3.125
        )

    for p in ['AM_PK', 'Md_OP', 'PM_PK', 'Nt_OP']:
        # plot(
        #     inundation_w_geo, p, 'Road inundation<br>severity',
        #     'inundation'
        # )
        # plot(
        #     congestion_w_geo_d, p, 'Travel time<br>increase',
        #     'congestion_d',"Cividis"
        # )
        plot(
            congestion_w_geo_change, p, 'Travel time<br>increase change',
            'congestion_change',"Electric", range_color=[-0.3, 0.7]
        )
    return


def calculate_mae(df, col_actual, col_predicted):
    import numpy as np
    mae = np.mean(np.abs(df[col_actual] - df[col_predicted]))
    return mae


def calculate_mape(df, col_actual, col_predicted):
    import numpy as np
    mape = np.mean(np.abs((df[col_actual] - df[col_predicted]) / df[col_actual])) * 100
    return mape


def calculate_rmse(df, col_actual, col_predicted):
    import numpy as np
    rmse = np.sqrt(np.mean((df[col_actual] - df[col_predicted]) ** 2))
    return rmse


def calculate_bias(df, col_actual, col_predicted):
    import numpy as np
    bias = np.mean(df[col_predicted] - df[col_actual])
    return bias

