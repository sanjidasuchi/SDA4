# @title FindMyRoute: Optimised Tourist Path Discovery - Google Colab Main Code

# --- Section 1: Verify Imports and Define Core Utility Functions ---
# This section attempts imports and defines all the functions.

print("\n--- Section 1: Verifying Imports and Defining Core Utility Functions ---")

# Verify imports after restart
try:
    import osmnx as ox 
    import networkx as nx
    import folium
    import math # For the improved lon_buffer_deg calculation
    print("All core libraries (osmnx, networkx, folium, math) imported successfully.")
except ImportError as e:
    print(f"ERROR: Failed to import a required library: {e}")
    print("Please ensure you ran the installation cell (Section 0) and restarted the runtime.")
    # Exit here if imports fail, as subsequent code won't work
    raise

# ... (the rest of your functions remain the same) ...
def load_city_graph(place_name: str) -> nx.MultiDiGraph:
    """
    Loads a city's road network graph from OpenStreetMap using osmnx.

    Args:
        place_name (str): The name of the city or area to load (e.g., "Salzburg, Austria").

    Returns:
        networkx.MultiDiGraph: A MultiDiGraph object representing the road network.
                               Returns None if the graph cannot be loaded.
    """
    print(f"\nAttempting to load graph for: {place_name}")
    try:
        # Retrieve the street network graph
        graph = ox.graph_from_place(place_name, network_type="drive")
        print(f"Successfully loaded graph for {place_name}")
        return graph
    except Exception as e:
        print(f"Error loading graph for {place_name}: {e}")
        print("Possible issues: Incorrect place name, no internet, or OSMnx server issues.")
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
        print("Error: Graph is not loaded. Cannot find nearest node.")
        return None
    try:
        # Find the nearest node to the given coordinates
        # osmnx.distance.nearest_nodes expects (longitude, latitude)
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
        print("Error: Graph is not loaded. Cannot calculate path.")
        return float('inf')

    # Convert coordinates to graph nodes
    origin_node = find_nearest_node(graph, origin_coords[0], origin_coords[1])
    destination_node = find_nearest_node(graph, destination_coords[0], destination_coords[1])

    if origin_node is None or destination_node is None:
        print("Could not find nearest nodes for path calculation.")
        return float('inf')

    print(f"\nCalculating shortest path from {origin_node} to {destination_node} (weight: {weight})...")
    try:
        # For travel time, ensure 'travel_time' attribute exists or calculate it
        if weight == 'travel_time':
            # Check if travel_time exists on at least one edge. If not, add it.
            # Avoids recalculating if already present.
            if not any('travel_time' in data for u, v, k, data in graph.edges(data=True, keys=True)):
                print("Adding edge speeds and travel times to graph...")
                graph = ox.speed.add_edge_speeds(graph)
                graph = ox.speed.add_travel_times(graph)
            else:
                print("Travel times already present on graph edges.")

        # Use Dijkstra's algorithm to find the shortest path
        path_length = nx.shortest_path_length(graph, source=origin_node, target=destination_node, weight=weight)
        unit = "meters" if weight == 'length' else "seconds"
        print(f"Shortest path length ({weight}) from {origin_node} to {destination_node}: {path_length:.2f} {unit}")
        return path_length
    except nx.NetworkXNoPath:
        print(f"No path found between {origin_node} and {destination_node} using '{weight}' weight.")
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
    center_lat, center_lon = 47.8095, 13.0550 # Default Salzburg coordinates (current location)

    if route_nodes:
        # Calculate approximate center from route nodes
        route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route_nodes]
        if route_coords: # Ensure route_coords is not empty
            center_lat = sum(coord[0] for coord in route_coords) / len(route_coords)
            center_lon = sum(coord[1] for coord in route_coords) / len(route_coords)
    elif start_point:
        center_lat, center_lon = start_point

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add the optimized route to the map
    if route_nodes:
        print("Plotting route on map...")
        try:
            # osmnx.plot_route_folium is the easiest way to plot a route
            route_map = ox.plot_route_folium(graph, route_nodes, route_map=m, tiles="OpenStreetMap",
                                             color="#cc0000", weight=5, opacity=0.7)
            m = route_map
        except Exception as e:
            print(f"Warning: Could not plot route directly with osmnx ({e}). Attempting to plot segments.")
            # Fallback: plot individual edges of the route if direct plotting fails
            for i in range(len(route_nodes) - 1):
                u = route_nodes[i]
                v = route_nodes[i+1]

                # Ensure nodes exist in the graph before trying to access their data
                if u not in graph.nodes or v not in graph.nodes:
                    print(f"Skipping segment from node {u} to {v} as one or both nodes not found in graph.")
                    continue

                # Get edge data (handling MultiDiGraph with multiple edges between u and v)
                edge_geometries = []
                if graph.has_edge(u, v):
                    for k, data in graph.get_edge_data(u, v).items():
                        if 'geometry' in data and data['geometry'] is not None:
                            edge_geometries.append(data['geometry'])

                if edge_geometries:
                    # Plot all geometries found for the segment
                    for geom in edge_geometries:
                        try:
                            # Shapely LineString.coords gives (x, y) which is (lon, lat)
                            folium.PolyLine(
                                locations=[(lat, lon) for lon, lat in geom.coords],
                                color="blue", weight=5, opacity=0.7
                            ).add_to(m)
                        except Exception as poly_e:
                            print(f"Error plotting PolyLine for segment geometry between {u} and {v}: {poly_e}")
                else:
                    # Fallback to straight line if no geometry or no direct edge found
                    start_lat, start_lon = graph.nodes[u]['y'], graph.nodes[u]['x']
                    end_lat, end_lon = graph.nodes[v]['y'], graph.nodes[v]['x']
                    folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)],
                                    color="blue", weight=5, opacity=0.7).add_to(m)

    # Add attractions markers
    if attractions:
        print("Adding attraction markers...")
        for name, coords in attractions.items():
            folium.Marker(
                location=coords,
                popup=f"<b>Attraction:</b> {name}",
                icon=folium.Icon(color="darkblue", icon="info-sign")
            ).add_to(m)

    # Add POI markers
    if pois:
        print("Adding POI markers...")
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
        print("Adding start point marker...")
        folium.Marker(
            location=start_point,
            popup="<b>Start Point</b>",
            icon=folium.Icon(color="red", icon="play")
        ).add_to(m)
    if end_point:
        print("Adding end point marker...")
        folium.Marker(
            location=end_point,
            popup="<b>End Point</b>",
            icon=folium.Icon(color="purple", icon="stop")
        ).add_to(m)

    print("Map creation process completed.")
    return m

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
    if not route_nodes or graph is None:
        print("Cannot find POIs: no route nodes or graph is not loaded.")
        return {}

    # Get the bounding box of the route
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')

    # Calculate bounding box from route nodes
    for node_id in route_nodes:
        if node_id in graph.nodes:
            node = graph.nodes[node_id]
            lat, lon = node['y'], node['x']
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
        else:
            print(f"Warning: Node {node_id} not found in graph for bounding box calculation. Skipping.")


    # Expand the bounding box by a buffer (rough approximation for A4)
    # This calculation uses a more robust approach for longitude buffer
    lat_buffer_deg = buffer_meters / 111139.0 # approx meters per degree latitude

    # Calculate average latitude for more accurate longitude degree conversion
    avg_lat_rad = (min_lat + max_lat) / 2 * (math.pi / 180.0) # Convert to radians
    # Handle cases where cos(avg_lat_rad) might be zero or near zero (e.g., at poles)
    if abs(math.cos(avg_lat_rad)) < 1e-6: # If close to pole (or equator if lat is 0), use a reasonable default
        lon_buffer_deg = buffer_meters / 111139.0
    else:
        lon_buffer_deg = buffer_meters / (111139.0 * math.cos(avg_lat_rad))

    north = max_lat + lat_buffer_deg
    south = min_lat - lat_buffer_deg
    east = max_lon + lon_buffer_deg
    west = min_lon - lon_buffer_deg

    print(f"\nSearching for POIs within approximate bounding box: ({south:.4f}, {west:.4f}, {north:.4f}, {east:.4f}) with tags {tags}...")

    pois = {}
    try:
        # Use osmnx to get features (POIs) within the expanded bounding box
        gdf_pois = ox.features.features_from_bbox(north, south, east, west, tags)

        # Filter for unique POIs and extract name and coordinates
        if not gdf_pois.empty:
            gdf_pois = gdf_pois.drop_duplicates(subset=['geometry']) # Avoid duplicate entries for same location

            for _, row in gdf_pois.iterrows():
                # Only consider features with a name and not internal OSM ways
                # Check for `name` existence and if it's not empty/whitespace and not an internal OSM way ID
                if 'name' in row and isinstance(row['name'], str) and row['name'].strip() and not row['name'].startswith('way/'):
                    # Prioritize point geometry, otherwise use centroid for polygons
                    if row.geometry.geom_type == 'Point':
                        pois[row['name']] = (row.geometry.y, row.geometry.x)
                    elif row.geometry.geom_type in ['Polygon', 'MultiPolygon'] and row.geometry.centroid:
                         pois[row['name']] = (row.geometry.centroid.y, row.geometry.centroid.x)
            print(f"Found {len(pois)} unique POIs near the route.")
        else:
            print("No features found in the bounding box for the specified tags.")

    except Exception as e:
        print(f"Error fetching POIs: {e}")
        # Specific check if no features were found (common osmnx error)
        if "No features match" in str(e) or "empty GeoDataFrame" in str(e):
             print("This might mean no POIs matching your tags exist in the queried area.")
    return pois

print("All utility functions are now defined and ready.")
print("--- End of Section 1 ---")


# --- Section 2: Main Demonstration Logic ---
# This section executes the functions to demonstrate the project's capabilities.

print("\n--- Section 2: Running Main Demonstration Logic ---")

## 2.1. Load City Road Network Graph

# Define the city for which to load the road network (current location)
city_name = "Salzburg, Austria"
print(f"Attempting to load road network for {city_name} (this may take a few moments)...")
G = load_city_graph(city_name)

if G is None:
    print("Graph loading failed. Aborting demonstration.")
    

## 2.2. Define Tourist Attractions and Points of Interest (POIs)

# Define some key attractions in Salzburg (Name: (Latitude, Longitude))
salzburg_attractions = {
    "Hohensalzburg Fortress": (47.7946, 13.0470),
    "Mozart's Birthplace": (47.8009, 13.0460),
    "Mirabell Palace": (47.8070, 13.0430),
    "Salzburg Cathedral": (47.7997, 13.0477),
    "Getreidegasse": (47.8000, 13.0450)
}

# Define a start and end point for a hypothetical route
# Using Getreidegasse as start and Mirabell Palace as end
start_coords = salzburg_attractions["Getreidegasse"]
end_coords = salzburg_attractions["Mirabell Palace"]

# Define OSM tags for POIs we are interested in (e.g., cafes, parks)
poi_tags = {
    'amenity': ['cafe', 'restaurant', 'pub', 'bench'],
    'leisure': ['park', 'garden', 'playground'],
    'shop': ['bakery', 'convenience']
}

print("\nAttractions and POI tags defined for demonstration.")

## 2.3. Calculate Shortest Path Between Two Points

print("\n--- Calculating shortest paths between selected attractions ---")

# Example: Calculate shortest distance between Hohensalzburg Fortress and Mirabell Palace
fortress_coords = salzburg_attractions["Hohensalzburg Fortress"]
mirabell_coords = salzburg_attractions["Mirabell Palace"]

# Calculate path based on physical length (meters)
distance_meters = calculate_shortest_path_length(G, fortress_coords, mirabell_coords, weight='length')
print(f"Distance between Fortress and Mirabell Palace: {distance_meters:.2f} meters")

# Calculate path based on estimated travel time (seconds)
travel_time_seconds = calculate_shortest_path_length(G, fortress_coords, mirabell_coords, weight='travel_time')
# Convert seconds to minutes for better readability
if travel_time_seconds != float('inf'):
    travel_time_minutes = travel_time_seconds / 60
    print(f"Estimated travel time between Fortress and Mirabell Palace: {travel_time_seconds:.2f} seconds ({travel_time_minutes:.2f} minutes)")
else:
    print("Could not calculate estimated travel time.")


## 2.4. Simplified Route Generation & POI Discovery (A4 Placeholder)

print("\n--- Generating a placeholder route and discovering nearby POIs ---")

# For A4, we simulate an 'optimized route' by finding a direct shortest path
# between our defined start and end coordinates.
start_node = find_nearest_node(G, start_coords[0], start_coords[1])
end_node = find_nearest_node(G, end_coords[0], end_coords[1])

route_nodes_for_viz = []
if start_node and end_node:
    try:
        # Use 'travel_time' for path finding to align with optimization goals
        # Ensure 'travel_time' attributes are on edges before trying to pathfind by it
        if not any('travel_time' in data for u, v, k, data in G.edges(data=True, keys=True)):
            print("Adding edge speeds and travel times for route generation...")
            G = ox.speed.add_edge_speeds(G)
            G = ox.speed.add_travel_times(G)
        route_nodes_for_viz = nx.shortest_path(G, source=start_node, target=end_node, weight='travel_time')
        print(f"Generated a direct path between start and end points with {len(route_nodes_for_viz)} nodes.")
    except nx.NetworkXNoPath:
        print("No direct path found between start and end points for visualization (check connectivity).")
    except Exception as e:
        print(f"Error generating route for visualization: {e}")
else:
    print("Start or end node not found. Cannot generate route.")

# Discover POIs near this simplified route
# Using a buffer of 500 meters around the route for POI search
discovered_pois = {}
if route_nodes_for_viz:
    print(f"\nSearching for POIs around the generated route with a {500}m buffer...")
    discovered_pois = find_pois_near_route(G, route_nodes_for_viz, tags=poi_tags, buffer_meters=500)
    print(f"Total discovered POIs: {len(discovered_pois)}")
else:
    print("No route available to find POIs near.")


## 2.5. Visualize the Route and POIs on an Interactive Map

print("\n--- Creating interactive map visualization ---")

# Create the interactive map
tour_map = visualize_route_on_map(
    graph=G,
    route_nodes=route_nodes_for_viz,
    attractions=salzburg_attractions,
    pois=discovered_pois,
    start_point=start_coords,
    end_point=end_coords
)

# Display the map. In Google Colab, this will render the interactive map below the cell.
print("\nDisplaying the interactive map. Scroll down to view the map output.")
tour_map

print("\n--- End of Section 2 ---")
print("\nFindMyRoute Demonstration Complete!")
