import networkx as nx
import arcpy
import os


def build_feature_dataset_arcgis(geodatabase_addr, road_seg_gdf, spatial_ref='./data/roads/4326.prj'):
    """
    Build a feature dataset in an ArcGIS geodatabase from a GeoDataFrame of road segments.

    Args:
        geodatabase_addr (str): Path to the ArcGIS geodatabase.
        road_seg_gdf (GeoDataFrame): Road segment data.
        spatial_ref (str, optional): Path to the spatial reference file. Defaults to './data/roads/4326.prj'.
    """
    # geodatabase should be pre-built from ArcGIS pro manually
    # road segment data should be in geojson format

    road_seg_gdf.to_file("road_seg_temp.geojson", driver="GeoJSON")
    arcpy.conversion.JSONToFeatures(  # load road segments as feature class
        "road_seg_temp.geojson",
        f'{geodatabase_addr}/road_segment', 'POLYLINE',
    )
    if os.path.exists("road_seg_temp.geojson"):
        os.remove("road_seg_temp.geojson")

    arcpy.management.CreateFeatureDataset(  # create an empty feature dataset to contain data for network
        geodatabase_addr,
        'road_network', spatial_ref
    )
    arcpy.conversion.FeatureClassToFeatureClass(  # put road feature class into feature dataset
        f'{geodatabase_addr}/road_segment',
        f'{geodatabase_addr}/road_network', 'road_seg'
    )
    return


def build_network_dataset_arcgis(geodatabase_addr):
    """
    Create and build a network dataset in an ArcGIS geodatabase.

    Args:
        geodatabase_addr (str): Path to the ArcGIS geodatabase.
    """
    arcpy.na.CreateNetworkDataset(  # create network dataset
        f'{geodatabase_addr}/road_network',
        "road_nd", 'road_seg',
    )
    arcpy.BuildNetwork_na(f'{geodatabase_addr}/road_network/road_nd')  # build network
    return


def _adapt_2_turn_rules_arcgis(rows_4_insert_list, road_seg_id, road_segment):
    """
    Adapt turn restriction rows for ArcGIS by handling interior and add-on roads.

    Args:
        rows_4_insert_list (list): List of rows to insert for turn restrictions.
        road_seg_id (int): Road segment ID.
        road_segment (GeoDataFrame): Road segment data.

    Returns:
        list: Updated list of rows for ArcGIS turn rules.
    """
    add_on_list = []
    interior_list = []
    for row in rows_4_insert_list:
        all_roads = row[2::3]
        interior_roads = [i for i in all_roads[1:] if i in all_roads[:-1]]
        interior_list.append(interior_roads)

    updated_rows_4_insert_list = []
    for row_index, row in enumerate(rows_4_insert_list):
        updated_row = row.copy()

        first_road = row[2]
        if first_road in [i for l in interior_list for i in l]:
            first_road_begin_node = road_segment.iloc[first_road - 1]['u']
            first_road_osm = road_segment.iloc[first_road - 1]['osmid']
            roads_same_osm = road_segment[road_segment['osmid'] == first_road_osm]
            preceding_road = roads_same_osm[roads_same_osm['v'] == first_road_begin_node]
            assert len(preceding_road) <= 1, 'Surplus preceding roads.'
            if len(preceding_road) == 0:
                # warnings.warn('No preceding roads with the same osmid.')
                """
                    Backup plan, attach it if there is only one preceding road.
                    It works because although a road can have multiple roads connected
                    it is likely that only one connected road is two-direction.
                    Rest of the roads are not real connected.
                """
                preceding_road_2 = road_segment[road_segment['v'] == first_road_begin_node]
                if len(preceding_road_2) == 1:
                    obj_id = preceding_road_2.index.values[0] + 1
                    if [row_index, obj_id - 1] not in add_on_list:
                        updated_row = row[:1] + [road_seg_id, obj_id, 0.5] + row[1:]
                        add_on_list.append([row_index, obj_id - 1])
            if len(preceding_road) == 1:
                obj_id = preceding_road.index.values[0] + 1
                if [row_index, obj_id - 1] not in add_on_list:
                    updated_row = row[:1] + [road_seg_id, obj_id, 0.5] + row[1:]
                    add_on_list.append([row_index, obj_id - 1])

        last_road = row[2::3][-1]
        if last_road in [i for l in interior_list for i in l]:
            last_road_end_node = road_segment.iloc[last_road - 1]['v']
            last_road_osm = road_segment.iloc[last_road - 1]['osmid']
            roads_same_osm = road_segment[road_segment['osmid'] == last_road_osm]
            following_road = roads_same_osm[roads_same_osm['u'] == last_road_end_node]
            assert len(following_road) <= 1, 'Surplus preceding roads.'
            if len(following_road) == 0:
                # warnings.warn('No following roads with the same osmid.')
                """
                    Backup plan. Explanation above.
                """
                following_road_2 = road_segment[road_segment['v'] == last_road_end_node]
                if len(following_road_2) == 1:
                    obj_id = following_road_2.index.values[0] + 1
                    if [row_index, obj_id - 1] not in add_on_list:
                        updated_row = row[:1] + [road_seg_id, obj_id, 0.5] + row[1:]
                        add_on_list.append([row_index, obj_id - 1])
            if len(following_road) == 1:
                obj_id = following_road.index.values[0] + 1
                if [row_index, obj_id - 1] not in add_on_list:
                    updated_row = row + [road_seg_id, obj_id, 0.5]
                    add_on_list.append([row_index, obj_id - 1])

        updated_rows_4_insert_list.append(updated_row)
    rows_4_insert_list = updated_rows_4_insert_list.copy()

    return rows_4_insert_list


def _insert_turn_rows_arcgis(addr_fd, turn_name, rows_4_insert_list, fields, road_segment):
    """
    Insert turn restriction rows into an ArcGIS feature dataset.

    Args:
        addr_fd (str): Path to the feature dataset.
        turn_name (str): Name of the turn restriction table.
        rows_4_insert_list (list): List of rows to insert.
        fields (list): List of field names.
        road_segment (GeoDataFrame): Road segment data.
    """
    import warnings
    # make up for the none
    updated_rows_4_insert_list = []
    for row in rows_4_insert_list:
        assert len(row) <= len(fields) - 1, 'Field length is not enough.'
        row = row + [None] * (len(fields) - len(row) - 1)
        updated_rows_4_insert_list.append(row)
    rows_4_insert_list = updated_rows_4_insert_list.copy()

    # insert
    cursor = arcpy.da.InsertCursor(
        f'{addr_fd}/{turn_name}',
        fields,
    )
    for row in rows_4_insert_list:
        # build geo
        geo_row_object_id = row[2::3]
        geo_row_index = [i - 1 for i in geo_row_object_id if i is not None]

        points_with_sequences = []
        for line in road_segment.iloc[geo_row_index]['geometry'].to_list():
            for point in line.coords:
                points_with_sequences.append(point)
        arcpy_points_with_sequences = [arcpy.Point(i[0], i[1]) for i in points_with_sequences]
        polyline = arcpy.Polyline(arcpy.Array(arcpy_points_with_sequences), arcpy.SpatialReference(4326))

        # insert
        try:
            cursor.insertRow([polyline] + row)
        except RuntimeError as e:
            if f"Objects in this class cannot be updated outside an edit session [{turn_name}]" in str(e):
                raise RuntimeError(
                    f"""
                        [{turn_name}] has been attached to the network dataset. Move to 
                        the user interface. Unselect [{turn_name}] in the Source Setting 
                        panel of the Network Dataset Properties. Then re-run the program.
                    """
                )
            else:
                raise
    del cursor
    warnings.warn('Must add turn restrictions to the network dataset manually, and build network again.')
    return


def add_turn_restriction_arcgis(
        addr_fd, fields, turn_name, turn_restriction, road_segment, wrong_restriction_list,
):
    """
    Add turn restrictions to an ArcGIS network dataset.

    Args:
        addr_fd (str): Path to the feature dataset.
        fields (list): List of field names.
        turn_name (str): Name of the turn restriction table.
        turn_restriction (DataFrame): Turn restriction data.
        road_segment (GeoDataFrame): Road segment data.
        wrong_restriction_list (list): List of restriction IDs to skip.
    """

    road_seg_id = arcpy.Describe(f'{addr_fd}/road_seg').DSID
    rows_4_insert_list = []
    for i, row in turn_restriction.iterrows():
        if row['rel'] in wrong_restriction_list:
            continue

        # get first intersection and first road
        if (len(row['via']) == 0) and (len(row['via_node']) == 1):
            # 'via' is a node
            intersection_id = row['via_node'][0]
        elif (len(row['via']) > 0) and (len(row['via_node']) == 0):
            # 'via' is roads
            from_end = road_segment[road_segment['osmid'] == int(row['from'])]['v'].to_list()
            via_start_ll = []
            for v in row['via']:
                via_start = road_segment[road_segment['osmid'] == int(v)]['u'].to_list()
                via_start_ll.append(via_start)
            via_start_l = [i for l in via_start_ll for i in l]
            intersection = [i for i in from_end if i in via_start_l]
            assert len(intersection) == 1, 'Surplus intersection or no intersection.'
            intersection_id = intersection[0]
        else:
            raise ValueError('Unexpected case.')

        applicable_roads = road_segment[road_segment['osmid'] == int(row['from'])]
        assert len(applicable_roads) >= 1
        correct_road = applicable_roads[applicable_roads['v'] == int(intersection_id)]
        assert len(correct_road) == 1
        fid_1 = correct_road.index.values[0] + 1

        # if 'via' is ways, process ways before attach 'to'
        def get_next_edge(v_l, inter_id):
            for vv in v_l:
                ap_roads = road_segment[road_segment['osmid'] == int(vv)]
                assert len(ap_roads) >= 1
                u_l = ap_roads['u'].to_list()

                if inter_id not in u_l:
                    continue

                cor_road = ap_roads[ap_roads['u'] == int(inter_id)]
                assert len(cor_road) == 1
                return cor_road.index.values[0] + 1, vv, cor_road
        row_4_insert_via_part = []

        if (len(row['via']) > 0) and (len(row['via_node']) == 0):
            fid_list = []
            v_list = row['via'].copy()
            while len(v_list) != 0:
                fid, v, correct_road = get_next_edge(v_list, intersection_id)
                fid_list.append(fid)
                v_list.remove(v)
                intersection_id = correct_road['v'].iloc[0]

            row_4_insert_via_part = [[road_seg_id, f, 0.5] for f in fid_list]

        # get 'to'
        applicable_roads = road_segment[road_segment['osmid'] == int(row['to'])]
        assert len(applicable_roads) >= 1
        correct_road = applicable_roads[applicable_roads['u'] == int(intersection_id)]
        assert len(correct_road) == 1
        fid_to = correct_road.index.values[0] + 1

        # row for insert
        row_4_insert = (
                ['Y', road_seg_id, fid_1, 0.5]
                + [i for l in row_4_insert_via_part for i in l]
                + [road_seg_id, fid_to, 0.5]
        )
        row_4_insert = row_4_insert
        rows_4_insert_list.append(row_4_insert)

    # The first and last edge cannot also be the interior edge of other restrictions (from ArcGIS)
    rows_4_insert_list = _adapt_2_turn_rules_arcgis(rows_4_insert_list, road_seg_id, road_segment)

    # remove duplicates
    rows_4_insert_list = [list(l) for l in set(tuple(l) for l in rows_4_insert_list)]

    # final insert
    _insert_turn_rows_arcgis(addr_fd, turn_name, rows_4_insert_list, fields, road_segment)

    return




