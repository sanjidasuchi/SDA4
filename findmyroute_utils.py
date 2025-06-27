# findmyroute_utils.py

"""
This module provides utility functions for the FindMyRoute project,
focusing on geospatial data acquisition, basic routing, and visualization
using osmnx, networkx, and folium.
"""

import osmnx as ox
import networkx as nx
import folium

def load_city_graph(place_name: str) -> nx.MultiDiGraph:
    """
    Loads a city's road network graph from OpenStreetMap using osmnx.

    Args:
        place_name (str): The name of the city or area to load (e.g., "Salzburg, Austria").

    Returns:
        networkx.MultiDiGraph: A MultiDiGraph object representing the road network.
                               Returns None if the graph cannot be loaded.
    """
    try:
        # Retrieve the street network graph
        graph = ox.graph_from_place(place_name, network_type="drive")
        print(f"Successfully loaded graph for {place_name}")
        return graph
    except Exception as e:
        print(f"Error loading graph for {place_name}: {e}")
        return None

def find_nearest_node(graph: nx.MultiDiGraph, latitude: float, longitude: float) -> int:
    """
    Finds the nearest graph node to a given latitude and longitude.

    Args:
        graph (networkx.MultiDiGraph): The road network graph.
        latitude (float): The latitude of the point.
        longitude (float): The longitude of the point.

    Returns:
        int: The ID of the nearest node in the graph. Returns None if the graph is invalid.
    """
    if graph is None:
        print("Error: Graph is not loaded.")
        return None
    try:
        # Find the nearest node to the given coordinates
        node = ox.distance.nearest_nodes(graph, longitude, latitude)
        print(f"Found nearest node {node} for coordinates ({latitude}, {longitude})")
        return node
    except Exception as e:
        print(f"Error finding nearest node for ({latitude}, {longitude}): {e}")
        return None

def calculate_shortest_path_length(graph: nx.MultiDiGraph, origin_coords: tuple, destination_coords: tuple, weight: str = 'length') -> float:
    """
    Calculates the length of the shortest path between two geographic points
    on the road network graph.

    Args:
        graph (networkx.MultiDiGraph): The road network graph.
        origin_coords (tuple): A tuple (latitude, longitude) for the origin point.
        destination_coords (tuple): A tuple (latitude, longitude) for the destination point.
        weight (str): The edge attribute to use as a weight for path calculation
                      (e.g., 'length' for distance in meters, 'travel_time' for time in seconds).

    Returns:
        float: The length of the shortest path in units of the specified weight.
               Returns float('inf') if no path exists or an error occurs.
    """
    if graph is None:
        print("Error: Graph is not loaded.")
        return float('inf')

    # Convert coordinates to graph nodes
    origin_node = find_nearest_node(graph, origin_coords[0], origin_coords[1])
    destination_node = find_nearest_node(graph, destination_coords[0], destination_coords[1])

    if origin_node is None or destination_node is None:
        print("Could not find nearest nodes for path calculation.")
        return float('inf')

    try:
        # Calculate the shortest path length
        # For travel time, ensure 'travel_time' attribute exists or calculate it
        if weight == 'travel_time' and 'travel_time' not in graph.edges(data=True).__next__()[2]:
            graph = ox.speed.add_edge_speeds(graph)
            graph = ox.speed.add_travel_times(graph)

        # Use Dijkstra's algorithm to find the shortest path
        path_length = nx.shortest_path_length(graph, source=origin_node, target=destination_node, weight=weight)
        print(f"Shortest path length ({weight}) from {origin_node} to {destination_node}: {path_length:.2f}")
        return path_length
    except nx.NetworkXNoPath:
        print(f"No path found between {origin_node} and {destination_node}.")
        return float('inf')
    except Exception as e:
        print(f"Error calculating shortest path: {e}")
        return float('inf')

def visualize_route_on_map(graph: nx.MultiDiGraph, route_nodes: list, attractions: dict = None, pois: dict = None, start_point: tuple = None, end_point: tuple = None) -> folium.Map:
    """
    Visualizes a given route on an interactive Folium map, optionally adding attractions and POIs.

    Args:
        graph (networkx.MultiDiGraph): The road network graph.
        route_nodes (list): A list of node IDs representing the optimized route.
        attractions (dict, optional): A dictionary of attractions {name: (lat, lon)}. Defaults to None.
        pois (dict, optional): A dictionary of POIs {name: (lat, lon)}. Defaults to None.
        start_point (tuple, optional): (latitude, longitude) of the start point. Defaults to None.
        end_point (tuple, optional): (latitude, longitude) of the end point. Defaults to None.

    Returns:
        folium.Map: An interactive Folium map object. Returns None if the graph is invalid.
    """
    if graph is None:
        print("Error: Graph is not loaded for visualization.")
        return None

    # Get the geographic centroid of the route for map centering
    if route_nodes:
        # Get coordinates for the route nodes
        route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route_nodes]
        # Calculate approximate center
        center_lat = sum(coord[0] for coord in route_coords) / len(route_coords)
        center_lon = sum(coord[1] for coord in route_coords) / len(route_coords)
    elif start_point:
        center_lat, center_lon = start_point
    else:
        # Fallback to a default center (e.g., Salzburg) if no route or start point is given
        center_lat, center_lon = 47.8095, 13.0550 # Salzburg coordinates

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add the optimized route to the map
    if route_nodes:
        # Extract geometry of the route if available, otherwise draw line segments
        try:
            # osmnx can plot the route directly
            route_map = ox.plot_route_folium(graph, route_nodes, route_map=m, tiles="OpenStreetMap",
                                             color="#cc0000", weight=5, opacity=0.7)
            m = route_map
        except Exception as e:
            print(f"Could not plot route directly with osmnx, plotting segments: {e}")
            # Fallback: plot individual edges of the route
            for i in range(len(route_nodes) - 1):
                start_node = route_nodes[i]
                end_node = route_nodes[i+1]
                try:
                    # Get the edge between the two nodes
                    # Need to handle potential multiple edges between nodes (MultiDiGraph)
                    # For simplicity, we'll take the first edge found
                    edge_data = graph.get_edge_data(start_node, end_node)
                    if edge_data:
                        # Assumes 'geometry' is present for complex paths or draws a straight line
                        if 'geometry' in edge_data[0]: # For MultiDiGraph, edge_data is a dict of dicts
                            folium.PolyLine(
                                locations=[(lat, lon) for lon, lat in edge_data[0]['geometry'].coords],
                                color="blue", weight=5, opacity=0.7
                            ).add_to(m)
                        else:
                             # Fallback to straight line if no geometry
                            start_lat, start_lon = graph.nodes[start_node]['y'], graph.nodes[start_node]['x']
                            end_lat, end_lon = graph.nodes[end_node]['y'], graph.nodes[end_node]['x']
                            folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)],
                                            color="blue", weight=5, opacity=0.7).add_to(m)
                except Exception as ex:
                    print(f"Error plotting segment between {start_node} and {end_node}: {ex}")


    # Add attractions markers
    if attractions:
        for name, coords in attractions.items():
            folium.Marker(
                location=coords,
                popup=f"<b>Attraction:</b> {name}",
                icon=folium.Icon(color="darkblue", icon="info-sign")
            ).add_to(m)

    # Add POI markers
    if pois:
        for name, coords in pois.items():
            folium.CircleMarker(
                location=coords,
                radius=5,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.6,
                popup=f"<b>POI:</b> {name}"
            ).add_to(m)

    # Add start and end points
    if start_point:
        folium.Marker(
            location=start_point,
            popup="<b>Start Point</b>",
            icon=folium.Icon(color="red", icon="play")
        ).add_to(m)
    if end_point:
        folium.Marker(
            location=end_point,
            popup="<b>End Point</b>",
            icon=folium.Icon(color="purple", icon="stop")
        ).add_to(m)
    
    print("Map created successfully with route and markers.")
    return m

# Example for POI filtering - simplified for A4
def find_pois_near_route(graph: nx.MultiDiGraph, route_nodes: list, tags: dict, buffer_meters: int = 500) -> dict:
    """
    A simplified function to find Points of Interest (POIs) near a given route.
    This is a basic implementation for the A4 and will be expanded in the final project.

    Args:
        graph (networkx.MultiDiGraph): The road network graph.
        route_nodes (list): A list of node IDs representing the optimized route.
        tags (dict): A dictionary of OSM tags to filter POIs (e.g., {'amenity': 'cafe'}).
        buffer_meters (int): The buffer distance around the route to search for POIs in meters.

    Returns:
        dict: A dictionary of identified POIs {name: (lat, lon)}.
    """
    if not route_nodes:
        return {}

    # Get the bounding box of the route
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')
    
    for node_id in route_nodes:
        node = graph.nodes[node_id]
        lat, lon = node['y'], node['x']
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
    
    # Expand the bounding box by a buffer (this is a rough approximation, not precise buffering)
    # A more precise method would involve projecting to a local UTM zone, buffering, then reprojecting.
    # For A4, we'll just expand the lat/lon bounds slightly.
    # Rough degree approximation for buffer_meters
    lat_buffer_deg = buffer_meters / 111139.0 # approx meters per degree latitude
    lon_buffer_deg = buffer_meters / (111139.0 * abs(max_lat if abs(max_lat) > abs(min_lat) else min_lat)) # meters per degree longitude varies with latitude
    if lon_buffer_deg > 1: # Avoid division by zero or huge buffer near equator
         lon_buffer_deg = buffer_meters / 111139.0 # Use a default if too close to poles or small area

    north = max_lat + lat_buffer_deg
    south = min_lat - lat_buffer_deg
    east = max_lon + lon_buffer_deg
    west = min_lon - lon_buffer_deg

    print(f"Searching for POIs within bounding box: ({south}, {west}, {north}, {east})")

    pois = {}
    try:
        # Use osmnx to get amenities within the expanded bounding box
        # This will be a more sophisticated spatial query in the final project
        gdf_pois = ox.features.features_from_bbox(north, south, east, west, tags)
        
        # Filter for unique POIs and extract name and coordinates
        for _, row in gdf_pois.iterrows():
            if 'name' in row and row['name'] and not row['name'].startswith('way/'): # Avoid unnamed ways
                # Check for point geometry (preferred) or polygon centroid
                if row.geometry.geom_type == 'Point':
                    pois[row['name']] = (row.geometry.y, row.geometry.x)
                elif row.geometry.geom_type in ['Polygon', 'MultiPolygon'] and row.geometry.centroid:
                     pois[row['name']] = (row.geometry.centroid.y, row.geometry.centroid.x)
        print(f"Found {len(pois)} POIs of type {tags} near the route.")
    except Exception as e:
        print(f"Error fetching POIs: {e}")
    
    return pois
