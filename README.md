# SDA4
Salzburg Museum Route Optimizer
This project focuses on leveraging geospatial data and optimization algorithms to help tourists and locals discover the most efficient way to visit museums in Salzburg, Austria. By integrating data from OpenStreetMap and applying a Traveling Salesman Problem (TSP) solver, it provides an optimized route designed to minimize travel time or distance.

Features
Museum Discovery: Identifies and retrieves information about museums within Salzburg using OpenStreetMap data via osmnx.

Route Optimization (TSP): Employs the Traveling Salesman Problem (TSP) algorithm (likely using Google OR-Tools) to calculate the shortest and most efficient sequence for visiting a user-defined set of museums.

Interactive Map Visualization: Presents the optimized museum route, along with individual museum locations, on an interactive map using folium, allowing for easy exploration and planning.

Estimated Travel Metrics: Provides insights into the estimated total travel time or distance for the optimized route.

Requirements
Python: Version 3.8 or higher.

Execution Environment: Designed to run seamlessly in Jupyter Notebook or Google Colab.

Internet Connection: Required for fetching OpenStreetMap data and displaying interactive maps.

How to Run
Prepare your environment:

For Google Colab: Open a new Colab notebook.

For Jupyter Notebook: Ensure you have Anaconda/Miniconda installed.

Install Dependencies: Run the following command in a cell (Colab) or your terminal (Jupyter) to install all necessary libraries. It's recommended to run this in a fresh environment or at the beginning of your Colab session, and potentially restart the runtime if using Colab.

!pip install osmnx==1.6.0 folium ortools geopy networkx # Adding networkx as it's a core dependency

Note: The specific osmnx==1.6.0 version is suggested for broader compatibility based on common Colab setups. If you encounter issues, newer versions might work, or specific dependency conflicts might need resolving.

Copy and Execute the Code:

Paste the entire project code (the Python script containing functions and main execution logic) into a cell (or multiple cells) in your Jupyter/Colab notebook.

Run the cells sequentially.

View Results: The interactive map displaying the optimized museum route and markers will be rendered directly within your notebook output.
