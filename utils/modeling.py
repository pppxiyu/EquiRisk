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
    service_area.outputType = arcpy.nax.ServiceAreaOutputType.PolygonsAndLines
    service_area.geometryAtCutoff = arcpy.nax.ServiceAreaPolygonCutoffGeometry.Disks
    service_area.geometryAtOverlap = arcpy.nax.ServiceAreaOverlapGeometry.Overlap

    fields = ["Name", "SHAPE@"]
    with service_area.insertCursor(arcpy.nax.ServiceAreaInputDataType.Facilities, fields) as cur:
        for i, row in rescue_station.iterrows():
            name, lat, lon = row['Number'], row['lat'], row['lon']
            cur.insertRow(([name, (lon, lat)]))

    return service_area


def run_service_area_analysis_arcgis(
        geodatabase_addr, fd_name, nd_name, nd_layer_name,
        rescue_station, cutoff=[300]
):
    arcpy.env.overwriteOutput = True
    arcpy.nax.MakeNetworkDatasetLayer(  # make layer
        f'{geodatabase_addr}/{fd_name}/{nd_name}', nd_layer_name
    )
    service_area_analyst = init_service_area_analysis_arcgis(
        nd_layer_name, rescue_station, cutoff
    )
    service_area_result = service_area_analyst.solve()
    assert service_area_result.solveSucceeded is True, 'Solving failed.'
    service_area_result.export(
        arcpy.nax.ServiceAreaOutputDataType.Polygons, f'{geodatabase_addr}/service_area_results',
    )


class RouteAnalysis:
    def __init__(self, incidents, rescue_station_col):
        self.incidents = incidents
        self.rescue_station_col = rescue_station_col
        self.num_sampling = None

    def init_route_analysis_arcgis(
            self, nd_layer_name, rescue_station, incidents,
    ):

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
        route.searchTolerance = 250
        route.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

        if self.num_sampling is not None:
            incidents = incidents.sample(n=self.num_sampling)
        rescue_station_involved = incidents[self.rescue_station_col].unique()
        rescue_station = rescue_station[rescue_station['Number'].isin(rescue_station_involved)]

        fields = ["RouteName", "Sequence", "SHAPE@"]
        count = 0
        with route.insertCursor(arcpy.nax.RouteInputDataType.Stops, fields) as cur:
            for i, res in rescue_station.iterrows():
                for j, inc in incidents[incidents[self.rescue_station_col] == res["Number"]].iterrows():
                    route_name = f'{res["Number"]}-{inc["incident_id"]}'
                    cur.insertRow(([
                        route_name, 1,
                        (res['lon'], res['lat'])
                    ]))
                    cur.insertRow(([
                        route_name, 2,
                        (inc['IncidentLon'], inc['IncidentLat']),
                    ]))
                    count += 1

        assert count == len(incidents)
        return route

    def run_route_analysis_arcgis(
            self,
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_station, roads,
    ):
        import warnings
        import json

        arcpy.env.overwriteOutput = True
        arcpy.nax.MakeNetworkDatasetLayer(  # make layer
            f'{geodatabase_addr}/{fd_name}/{nd_name}', nd_layer_name
        )

        period_list = list(self.incidents['period_label'].unique())

        inaccessible_route_normal = []
        if '' in period_list:
            incidents_select = self.incidents[self.incidents['period_label'] == '']
            route_analyst = self.init_route_analysis_arcgis(
                nd_layer_name, rescue_station, incidents_select,
            )
            route_result = route_analyst.solve()
            route_result.export(
                arcpy.nax.RouteOutputDataType.Routes, f'{geodatabase_addr}/route_results_normal',
            )
            if route_result.isPartialSolution:
                warnings.warn(
                    f"{len(incidents_select) - route_result.count(arcpy.nax.RouteOutputDataType.Routes)} "
                    f"routes are missing when do for normal.")
                error_list = route_result.solverMessages(arcpy.nax.MessageSeverity.All)
                inaccessible_route_normal = [e[1].split('"')[1] for e in error_list if e[1].startswith('No route for')]

        inaccessible_route_flood_list = {}
        if [i for i in period_list if i != ''] != []:
            cost_eval, restriction_eval = add_custom_edge_evaluator(
                'travel_time_s', 'inundation',
                f'{geodatabase_addr}/{fd_name}/{nd_name}',
                roads,
            )
            for label in [i for i in period_list if i != '']:
                incidents_select = self.incidents[self.incidents['period_label'] == label]
                if len(incidents_select) == 0:
                    continue

                cost_eval.label = label
                restriction_eval.label = label
                route_analyst = self.init_route_analysis_arcgis(
                    nd_layer_name, rescue_station, incidents_select,
                )
                route_result = route_analyst.solve()
                route_result.export(
                    arcpy.nax.RouteOutputDataType.Routes, f'{geodatabase_addr}/route_results_{label}',
                )
                if route_result.isPartialSolution:
                    warnings.warn(
                        f"{len(incidents_select) - route_result.count(arcpy.nax.RouteOutputDataType.Routes)} "
                        f"routes are missing for period {label}.")
                    error_list = route_result.solverMessages(arcpy.nax.MessageSeverity.All)
                    print(error_list)
                    inaccessible_route_flood_list[f'{label}'] = error_list

        inaccessible_route_save = {
            'normal': inaccessible_route_normal, 'flood': inaccessible_route_flood_list
        }
        with open('./data/incidents/inaccessible_route.json', 'w') as f:
            json.dump(inaccessible_route_save, f)


class EdgeCostCustomizer(arcpy.nax.AttributeEvaluator):
    def __init__(self, attributeName, roads, sourceNames=None):
        super(EdgeCostCustomizer, self).__init__(attributeName, sourceNames)
        self.roads = roads
        self.label = ''

    def edgeValue(self, edge: arcpy.nax.Edge):
        edge_num = self.networkQuery.sourceInfo(edge)[1]
        # if self.label == '':
        #     value = self.roads.loc[edge_num - 1, 'travel_time_s']
        # else:
        value = self.roads.loc[edge_num - 1, f'travel_time_s_{self.label}']
        return value


class EdgeRestrictionCustomizer(arcpy.nax.AttributeEvaluator):
    def __init__(self, attributeName, roads, sourceNames=None):
        super(EdgeRestrictionCustomizer, self).__init__(attributeName, sourceNames)
        self.roads = roads
        self.label = ''

    def edgeValue(self, edge: arcpy.nax.Edge):
        edge_num = self.networkQuery.sourceInfo(edge)[1]
        value = self.roads.loc[edge_num - 1, f'maxspeed_inundated_mile_{self.label}']
        if value <= 1e-3:
            return True
        else:
            return False


def add_custom_edge_evaluator(cost_attr_name, restriction_attr_name, nd_name, roads):
    cost_customizer = EdgeCostCustomizer(
        cost_attr_name, roads,
        ['road_seg'],
    )
    restriction_customizer = EdgeRestrictionCustomizer(
        restriction_attr_name, roads,
        ['road_seg'],
    )
    network_dataset = arcpy.nax.NetworkDataset(nd_name)
    network_dataset.customEvaluators = [cost_customizer, restriction_customizer]
    return cost_customizer, restriction_customizer

