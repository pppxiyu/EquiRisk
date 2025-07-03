import arcpy


def init_service_area_analysis_arcgis(nd_layer_name, rescue_station, cutoff_time: list):
    """
    Initialize ArcGIS Service Area analysis for rescue stations.
    
    Creates and configures a Service Area analysis object in ArcGIS Network Analyst.
    This function sets up the analysis parameters and adds rescue stations as facilities.
    
    Args:
        nd_layer_name (str): Name of the network dataset layer in ArcGIS
        rescue_station (pandas.DataFrame): DataFrame containing rescue station data with columns:
            - 'Number': Station identifier
            - 'lat': Latitude coordinate
            - 'lon': Longitude coordinate
        cutoff_time (list): List of time thresholds (in seconds) for service area calculation
    
    Returns:
        arcpy.nax.ServiceArea: Configured Service Area analysis object ready for solving
        
    Raises:
        ValueError: If network dataset doesn't have required travel modes configured
        
    Note:
        Requires manual setup of 'DriveTime' travel mode in ArcGIS Network Dataset properties.
        This cannot be done programmatically due to ArcGIS limitations.
        
    Dependencies:
        - config_vb.py: Uses configuration parameters for travel mode setup
        - utils/preprocess_station.py: Rescue station data format
    """
    
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
    """
    Execute complete Service Area analysis workflow in ArcGIS.
    
    Performs a full service area analysis including network layer creation, analysis setup,
    solving, and result export. This is a convenience function that orchestrates the entire
    service area analysis process.
    
    Args:
        geodatabase_addr (str): Path to the ArcGIS geodatabase
        fd_name (str): Feature dataset name within the geodatabase
        nd_name (str): Network dataset name within the feature dataset
        nd_layer_name (str): Name for the network dataset layer
        rescue_station (pandas.DataFrame): Rescue station data with location information
        cutoff (list, optional): Time thresholds for service area calculation. Defaults to [300].
    
    Returns:
        None: Results are exported to the geodatabase
        
    Raises:
        AssertionError: If the analysis fails to solve successfully
        
    Dependencies:
        - init_service_area_analysis_arcgis(): Called to set up the analysis
        - config_vb.py: Uses geodatabase and network dataset configuration
        - utils/preprocess_station.py: Rescue station data format
    """
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
    """
    Route analysis class for emergency response scenarios.
    
    This class handles route calculation between rescue stations and incident locations
    under both normal and flood conditions. It supports different operational modes
    and can handle custom edge evaluators for flood-specific network modifications.
    
    The class integrates with the main modeling workflow in main_modeling.py and
    supports various routing strategies (OP0-OP5) defined in the configuration.
    
    Attributes:
        incidents (pandas.DataFrame): Incident data with location and timing information
        rescue_station_col (str): Column name identifying the assigned rescue station
        num_sampling (int, optional): Number of incidents to sample for analysis
        mode_label (str): Label suffix for output files to distinguish different modes
    
    Dependencies:
        - config_vb.py: Configuration parameters and file paths
        - utils/preprocess_incidents.py: Incident data format and processing
        - utils/preprocess_station.py: Rescue station data format
        - utils/preprocess_roads.py: Road network data with flood information
        - main_modeling.py: Called from calculate_all_routes() function
    """
    
    def __init__(self, incidents, rescue_station_col, mode_label=''):
        """
        Initialize RouteAnalysis with incident data and configuration.
        
        Args:
            incidents (pandas.DataFrame): DataFrame containing incident data with columns:
                - 'incident_id': Unique incident identifier
                - 'IncidentLat': Incident latitude
                - 'IncidentLon': Incident longitude
                - 'period_label': Time period label for flood analysis
                - rescue_station_col: Column containing assigned rescue station
            rescue_station_col (str): Column name in incidents DataFrame that identifies
                the rescue station assigned to each incident
            mode_label (str, optional): Suffix for output files to distinguish different
                operational modes (e.g., '_o' for real origins, '_daily_c' for daily congestion)
        """
        self.incidents = incidents
        self.rescue_station_col = rescue_station_col
        self.num_sampling = None
        self.mode_label = mode_label

    def init_route_analysis_arcgis(
            self, nd_layer_name, rescue_station, incidents,
    ):
        """
        Initialize ArcGIS Route analysis for incident-response routing.
        
        Creates and configures a Route analysis object in ArcGIS Network Analyst.
        Sets up the analysis parameters and adds rescue stations and incidents as stops.
        
        Args:
            nd_layer_name (str): Name of the network dataset layer in ArcGIS
            rescue_station (pandas.DataFrame): Rescue station data with location information
            incidents (pandas.DataFrame): Incident data to be routed to/from rescue stations
        
        Returns:
            arcpy.nax.Route: Configured Route analysis object ready for solving
            
        Raises:
            ValueError: If network dataset doesn't have required travel modes configured
            AssertionError: If the number of created routes doesn't match incident count
            
        Note:
            Requires manual setup of 'DriveTime' travel mode in ArcGIS Network Dataset properties.
            Creates routes from rescue stations to incident locations with sequence numbers.
        """

        note = """
            MANUAL OPERATION: go to the network dataset under the feature dataset. Right click the nd, and 
            add a new travel mode called 'DriveTime' manually. This step must be finished in the UI, as the 
            travel mode attribute of nd is set as read-only. Esri does not support adding a travel mode from 
            scratch to nd yet. Ref: https://community.esri.com/t5/arcgis-network-analyst-questions/
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
            rescue_station, roads, if_do_normal=False, if_do_flood=True
    ):
        """
        Execute complete route analysis workflow for normal and flood conditions.
        
        Performs route analysis for incidents under different conditions:
        - Normal conditions: Baseline routing without flood modifications
        - Flood conditions: Routing with custom edge evaluators for flood-specific
          travel times and restrictions
        
        The function handles multiple time periods for flood analysis and exports
        results to the geodatabase. It also tracks inaccessible routes and saves
        error information for analysis.
        
        Args:
            geodatabase_addr (str): Path to the ArcGIS geodatabase
            fd_name (str): Feature dataset name within the geodatabase
            nd_name (str): Network dataset name within the feature dataset
            nd_layer_name (str): Name for the network dataset layer
            rescue_station (pandas.DataFrame): Rescue station data with location information
            roads (pandas.DataFrame): Road network data with flood-specific travel times
            if_do_normal (bool, optional): Whether to perform normal condition analysis. Defaults to False.
            if_do_flood (bool, optional): Whether to perform flood condition analysis. Defaults to True.
        
        Returns:
            None: Results are exported to the geodatabase and error logs are saved
            
        Dependencies:
            - init_route_analysis_arcgis(): Called to set up route analysis
            - add_custom_edge_evaluator(): Called to modify network for flood conditions
            - config_vb.py: Uses geodatabase paths and period configuration
            - utils/preprocess_incidents.py: Incident data with period labels
            - utils/preprocess_roads.py: Road data with flood-specific attributes
        """
        import warnings
        import json

        arcpy.env.overwriteOutput = True
        arcpy.nax.MakeNetworkDatasetLayer(  # make layer
            f'{geodatabase_addr}/{fd_name}/{nd_name}', nd_layer_name
        )

        period_list = list(self.incidents['period_label'].unique())

        inaccessible_route_normal = []
        if '' in period_list:
            if if_do_normal:
                incidents_select = self.incidents[self.incidents['period_label'] == '']
                route_analyst = self.init_route_analysis_arcgis(
                    nd_layer_name, rescue_station, incidents_select,
                )
                route_result = route_analyst.solve()
                route_result.export(
                    arcpy.nax.RouteOutputDataType.Routes, f'{geodatabase_addr}/route_results_normal{self.mode_label}',
                )
                if route_result.isPartialSolution:
                    warnings.warn(
                        f"{len(incidents_select) - route_result.count(arcpy.nax.RouteOutputDataType.Routes)} "
                        f"routes are missing when do for normal.")
                    error_list = route_result.solverMessages(arcpy.nax.MessageSeverity.All)
                    inaccessible_route_normal = [e[1].split('"')[1] for e in error_list if e[1].startswith('No route for')]

        if if_do_flood:
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
                        arcpy.nax.RouteOutputDataType.Routes, f'{geodatabase_addr}/route_results_{label}{self.mode_label}',
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
            with open(f'data/bch/incidents/inaccessible_route{self.mode_label}.json', 'w') as f:
                json.dump(inaccessible_route_save, f)


class ServiceAreaAnalysis:
    """
    Service area analysis class for emergency response coverage assessment.
    
    This class handles service area calculation around rescue stations to determine
    coverage areas under different conditions. It supports both normal and flood
    scenarios with custom edge evaluators for flood-specific network modifications.
    
    The class is used to assess emergency service coverage and identify areas that
    may become inaccessible during flood events.
    
    Attributes:
        travel_threshold (int): Time threshold (in seconds) for service area calculation
        incidents (pandas.DataFrame): Incident data used to determine which rescue stations to analyze
        rescue_station_col (str): Column name identifying the assigned rescue station
        num_sampling (int, optional): Number of incidents to sample for analysis
        mode_label (str): Label suffix for output files to distinguish different modes
    
    Dependencies:
        - config_vb.py: Configuration parameters including service_area_threshold
        - utils/preprocess_incidents.py: Incident data format and processing
        - utils/preprocess_station.py: Rescue station data format
        - utils/preprocess_roads.py: Road network data with flood information
        - main_modeling.py: Called from main modeling workflow
    """
    
    def __init__(self, travel_threshold, incidents, rescue_station_col, mode_label=''):
        """
        Initialize ServiceAreaAnalysis with threshold and incident data.
        
        Args:
            travel_threshold (int): Time threshold in seconds for service area calculation.
                Areas reachable within this time are considered within service coverage.
            incidents (pandas.DataFrame): DataFrame containing incident data used to
                determine which rescue stations to include in the analysis
            rescue_station_col (str): Column name in incidents DataFrame that identifies
                the rescue station assigned to each incident
            mode_label (str, optional): Suffix for output files to distinguish different
                operational modes
        """
        self.travel_threshold = travel_threshold
        self.incidents = incidents
        self.rescue_station_col = rescue_station_col
        self.num_sampling = None
        self.mode_label = mode_label

    def init_service_area_analysis_arcgis(
            self, nd_layer_name, rescue_station, incidents
    ):
        """
        Initialize ArcGIS Service Area analysis for rescue station coverage.
        
        Creates and configures a Service Area analysis object in ArcGIS Network Analyst.
        Sets up the analysis parameters and adds rescue stations as facilities.
        
        Args:
            nd_layer_name (str): Name of the network dataset layer in ArcGIS
            rescue_station (pandas.DataFrame): Rescue station data with location information
            incidents (pandas.DataFrame): Incident data used to filter rescue stations
        
        Returns:
            arcpy.nax.ServiceArea: Configured Service Area analysis object ready for solving
            
        Raises:
            ValueError: If network dataset doesn't have required travel modes configured
            AssertionError: If the number of added facilities doesn't match rescue station count
            
        Note:
            Requires manual setup of 'DriveTime' travel mode in ArcGIS Network Dataset properties.
            Only includes rescue stations that are involved in the incident dataset.
        """

        note = """
            MANUAL OPERATION: go to the network dataset under the feature dataset. Right click the nd, and 
            add a new travel mode called 'DriveTime' manually. This step must be finished in the UI, as the 
            travel mode attribute of nd is set as read-only. Esri does not support adding a travel mode from 
            scratch to nd yet. Ref: https://community.esri.com/t5/arcgis-network-analyst-questions/
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
        service_area.defaultImpedanceCutoffs = [self.travel_threshold]
        service_area.travelMode = arcpy.nax.GetTravelModes(nd_layer_name)['DriveTime']
        service_area.outputType = arcpy.nax.ServiceAreaOutputType.Polygons
        service_area.geometryAtOverlap = arcpy.nax.ServiceAreaOverlapGeometry.Dissolve
        service_area.searchTolerance = 250
        service_area.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

        if self.num_sampling is not None:
            incidents = incidents.sample(n=self.num_sampling)
        rescue_station_involved = incidents[self.rescue_station_col].unique()
        rescue_station = rescue_station[rescue_station['Number'].isin(rescue_station_involved)]

        fields = ["Name", "SHAPE@"]
        count = 0
        with service_area.insertCursor(arcpy.nax.ServiceAreaInputDataType.Facilities, fields) as cur:
            for i, res in rescue_station.iterrows():
                cur.insertRow(([f'{res["Number"]}', (res['lon'], res['lat'])]))
                count += 1
        assert count == len(rescue_station)
        return service_area

    def run_service_area_analysis_arcgis(
            self,
            geodatabase_addr, fd_name, nd_name, nd_layer_name,
            rescue_station, roads, if_do_normal=False, if_do_flood=True
    ):
        """
        Execute complete service area analysis workflow for normal and flood conditions.
        
        Performs service area analysis for rescue stations under different conditions:
        - Normal conditions: Baseline service areas without flood modifications
        - Flood conditions: Service areas with custom edge evaluators for flood-specific
          travel times and restrictions
        
        The function handles multiple time periods for flood analysis and exports
        results to the geodatabase. It also tracks facility failures and saves
        error information for analysis.
        
        Args:
            geodatabase_addr (str): Path to the ArcGIS geodatabase
            fd_name (str): Feature dataset name within the geodatabase
            nd_name (str): Network dataset name within the feature dataset
            nd_layer_name (str): Name for the network dataset layer
            rescue_station (pandas.DataFrame): Rescue station data with location information
            roads (pandas.DataFrame): Road network data with flood-specific travel times
            if_do_normal (bool, optional): Whether to perform normal condition analysis. Defaults to False.
            if_do_flood (bool, optional): Whether to perform flood condition analysis. Defaults to True.
        
        Returns:
            None: Results are exported to the geodatabase and error logs are saved
            
        Dependencies:
            - init_service_area_analysis_arcgis(): Called to set up service area analysis
            - add_custom_edge_evaluator(): Called to modify network for flood conditions
            - config_vb.py: Uses geodatabase paths and period configuration
            - utils/preprocess_incidents.py: Incident data with period labels
            - utils/preprocess_roads.py: Road data with flood-specific attributes
        """
        import warnings
        import json

        arcpy.env.overwriteOutput = True
        arcpy.nax.MakeNetworkDatasetLayer(  # make layer
            f'{geodatabase_addr}/{fd_name}/{nd_name}', nd_layer_name
        )

        period_list = list(self.incidents['period_label'].unique())

        failure_message_normal = []
        if '' in period_list:
            if if_do_normal:
                incidents_select = self.incidents[self.incidents['period_label'] == '']
                service_area_analyst = self.init_service_area_analysis_arcgis(
                    nd_layer_name, rescue_station, incidents_select,
                )
                service_area_result = service_area_analyst.solve()
                service_area_result.export(
                    arcpy.nax.ServiceAreaOutputDataType.Polygons, f'{geodatabase_addr}/service_area_results_normal{self.mode_label}',
                )
                if service_area_result.isPartialSolution:
                    error_list = service_area_result.solverMessages(arcpy.nax.MessageSeverity.All)
                    failure_message_normal = [e for e in error_list]

        if if_do_flood:
            failure_message_flood_list = {}
            if [i for i in period_list if i != ''] != []:
                cost_eval, restriction_eval = add_custom_edge_evaluator(
                    'travel_time_s', 'inundation',
                    f'{geodatabase_addr}/{fd_name}/{nd_name}',
                    roads,
                )
                for label in [i for i in period_list if i != '']:
                    cost_eval.label = label
                    restriction_eval.label = label
                    service_area_analyst = self.init_service_area_analysis_arcgis(
                        nd_layer_name, rescue_station, self.incidents,
                    )
                    service_area_result = service_area_analyst.solve()
                    service_area_result.export(
                        arcpy.nax.ServiceAreaOutputDataType.Polygons, f'{geodatabase_addr}/service_area_results_{label}{self.mode_label}',
                    )
                    if service_area_result.isPartialSolution:
                        error_list = service_area_result.solverMessages(arcpy.nax.MessageSeverity.All)
                        failure_message_flood_list[f'{label}'] = error_list

            inaccessible_route_save = {
                'normal': failure_message_normal, 'flood': failure_message_flood_list
            }
            with open(f'data/VB/incidents/failed_facility{self.mode_label}.json', 'w') as f:
                json.dump(inaccessible_route_save, f)


class EdgeCostCustomizer(arcpy.nax.AttributeEvaluator):
    """
    Custom edge cost evaluator for flood-specific travel time modifications.
    
    This class extends ArcGIS Network Analyst's AttributeEvaluator to provide
    dynamic travel time costs based on flood conditions. It reads flood-specific
    travel times from the road network data and applies them during network analysis.
    
    The evaluator is used in both route analysis and service area analysis to
    simulate realistic travel conditions during flood events.
    
    Attributes:
        roads (pandas.DataFrame): Road network data containing flood-specific travel times
        label (str): Time period label used to select appropriate travel time column
    
    Dependencies:
        - utils/preprocess_roads.py: Road data with flood-specific travel time columns
        - config_vb.py: Period configuration for flood analysis
    """
    
    def __init__(self, attributeName, roads, sourceNames=None):
        """
        Initialize EdgeCostCustomizer with road data.
        
        Args:
            attributeName (str): Name of the network attribute to evaluate
            roads (pandas.DataFrame): Road network data with flood-specific travel time columns
            sourceNames (list, optional): List of source feature class names. Defaults to None.
        """
        super(EdgeCostCustomizer, self).__init__(attributeName, sourceNames)
        self.roads = roads
        self.label = ''

    def edgeValue(self, edge: arcpy.nax.Edge):
        """
        Calculate the travel time cost for a network edge.
        
        Retrieves the flood-specific travel time for the given edge based on the
        current time period label. The method accesses the road data using the
        edge's source information.
        
        Args:
            edge (arcpy.nax.Edge): Network edge to evaluate
            
        Returns:
            float: Travel time in seconds for the edge under current flood conditions
            
        Note:
            The method expects road data to have columns named 'travel_time_s_{label}'
            where {label} corresponds to the time period being analyzed.
        """
        edge_num = self.networkQuery.sourceInfo(edge)[1]
        # if self.label == '':
        #     value = self.roads.loc[edge_num - 1, 'travel_time_s']
        # else:
        value = self.roads.loc[edge_num - 1, f'travel_time_s_{self.label}']
        return value


class EdgeRestrictionCustomizer(arcpy.nax.AttributeEvaluator):
    """
    Custom edge restriction evaluator for flood-specific road closures.
    
    This class extends ArcGIS Network Analyst's AttributeEvaluator to provide
    dynamic road restrictions based on flood conditions. It determines whether
    roads should be closed based on flood depth and speed restrictions.
    
    The evaluator is used in both route analysis and service area analysis to
    simulate road closures during flood events.
    
    Attributes:
        roads (pandas.DataFrame): Road network data containing flood-specific speed restrictions
        label (str): Time period label used to select appropriate restriction column
    
    Dependencies:
        - utils/preprocess_roads.py: Road data with flood-specific speed restriction columns
        - config_vb.py: Period configuration for flood analysis
    """
    
    def __init__(self, attributeName, roads, sourceNames=None):
        """
        Initialize EdgeRestrictionCustomizer with road data.
        
        Args:
            attributeName (str): Name of the network attribute to evaluate
            roads (pandas.DataFrame): Road network data with flood-specific speed restriction columns
            sourceNames (list, optional): List of source feature class names. Defaults to None.
        """
        super(EdgeRestrictionCustomizer, self).__init__(attributeName, sourceNames)
        self.roads = roads
        self.label = ''

    def edgeValue(self, edge: arcpy.nax.Edge):
        """
        Determine if a network edge should be restricted (closed) due to flooding.
        
        Checks the flood-specific speed restriction for the given edge. If the
        maximum speed under flood conditions is very low (≤ 0.001 mph), the road
        is considered closed and the method returns True to restrict it.
        
        Args:
            edge (arcpy.nax.Edge): Network edge to evaluate
            
        Returns:
            bool: True if the edge should be restricted (closed), False otherwise
            
        Note:
            The method expects road data to have columns named 'maxspeed_inundated_mile_{label}'
            where {label} corresponds to the time period being analyzed.
            Roads with speed ≤ 0.001 mph are considered impassable.
        """
        edge_num = self.networkQuery.sourceInfo(edge)[1]
        value = self.roads.loc[edge_num - 1, f'maxspeed_inundated_mile_{self.label}']
        if value <= 1e-3:
            return True
        else:
            return False


def add_custom_edge_evaluator(cost_attr_name, restriction_attr_name, nd_name, roads):
    """
    Add custom edge evaluators to a network dataset for flood analysis.
    
    Creates and configures custom edge evaluators for travel time costs and
    road restrictions based on flood conditions. These evaluators modify the
    network behavior during analysis to simulate realistic flood impacts.
    
    Args:
        cost_attr_name (str): Name of the cost attribute in the network dataset
        restriction_attr_name (str): Name of the restriction attribute in the network dataset
        nd_name (str): Path to the network dataset
        roads (pandas.DataFrame): Road network data with flood-specific attributes
    
    Returns:
        tuple: (EdgeCostCustomizer, EdgeRestrictionCustomizer) - The configured evaluators
        
    Dependencies:
        - EdgeCostCustomizer: Custom evaluator for travel time costs
        - EdgeRestrictionCustomizer: Custom evaluator for road restrictions
        - utils/preprocess_roads.py: Road data with flood-specific columns
        - config_vb.py: Network dataset configuration
        
    Note:
        The evaluators are applied to the network dataset and will be used in
        subsequent route and service area analyses. The label attribute of the
        evaluators should be set to the appropriate time period before analysis.
    """
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


