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
        width=800,
        height=600,
    )
    return fig


def scatter_demo_vs_error_w_period(df, col_demo, col_error, col_color, col_time, xaxis, yaxis):
    df[col_color] = df[col_color].astype(float)
    color_map = {row[col_color]: row[col_time] for index, row in df.iterrows()}
    fig = px.scatter(
        df, x=col_demo, y=col_error,
        color=col_color, color_continuous_scale='Emrld'
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
            tickmode='linear'
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


def scatter_demo_vs_error(df, col_demo, col_error, xaxis, yaxis):
    fig = px.scatter(
        df, x=col_demo, y=col_error,
    )
    fig.update_layout(
        xaxis=dict(
            title=xaxis,
            showline=True,
            linewidth=1.5,
            linecolor='black',
            showgrid=False,
            ticks='inside',
            nticks=5,
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
    fig.update_traces(
        marker=dict(
            size=25,
            line=dict(color='black', width=1),
            opacity=0.75,
            color=px.colors.sequential.Emrld[len(px.colors.sequential.Emrld) // 2],
        )
    )
    fig = layout(fig)
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



