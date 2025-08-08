import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import matplotlib.cm as cm
from streamlit_folium import st_folium
import folium
import openrouteservice
import streamlit.components.v1 as components
import json
import hashlib
import pickle
import os

# --------- JSON serialization helper ---------
def convert_np(o):
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    return str(o)

# --------- Hash helpers for caching ---------
def file_hash(file_bytes):
    # Hash file content for unique caching even if user uploads the same file again
    return hashlib.sha256(file_bytes).hexdigest()

def config_hash(*args):
    # Hash all arguments for caching configs/constraints/locations
    return hashlib.sha256("_".join([str(a) for a in args]).encode()).hexdigest()

st.set_page_config(page_title="Smart VRP/Beat Optimizer", layout="wide")
st.title("🚚 Smart Beat/Route Optimizer (Multi-Warehouse, Real Road Routing, Cached)")

# 0. API Keys
st.sidebar.header("🔑 API Keys (Required)")
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjZiOTQzYjM2YzBkMDQ2NmQ4ZmE4ZGQ5ODI5MTAwZWE5IiwiaCI6Im11cm11cjY0In0="
MAPBOX_TOKEN = "pk.eyJ1Ijoic2FpbWFyYW5pIiwiYSI6ImNtZHpxNGRsbDA3MnEyaXNicGgyeTVocjAifQ.Is217ZkO0dJuAGqpPFr9PQ"
if not ORS_API_KEY or not MAPBOX_TOKEN:
    st.warning("Please enter both ORS and Mapbox API keys in the sidebar.")
    st.stop()
client = openrouteservice.Client(key=ORS_API_KEY)

# 1. Upload Shop Data
st.header("1️⃣ Upload Shop Data")
uploaded = st.file_uploader("Upload CSV (columns: shop_id, latitude, longitude, address, sales)", type=["csv"])
if not uploaded:
    st.info("Upload your CSV to proceed.")
    st.stop()
file_bytes = uploaded.read()
uploaded.seek(0)
df = pd.read_csv(uploaded)
df['latitude'].replace(0, np.nan, inplace=True)
df['longitude'].replace(0, np.nan, inplace=True)
df.dropna(subset=['latitude', 'longitude'], inplace=True)
df.reset_index(drop=True, inplace=True)

st.write(f"Rows after dropna: {len(df)}")
st.write(df.head())

# 2. Warehouses and Constraints Setup
st.header("2️⃣ Warehouses & Beat Settings")
num_wh = st.number_input("How many warehouses?", min_value=1, max_value=5, value=1, step=1)
warehouse_settings = []

for i in range(int(num_wh)):
    with st.expander(f"🏭 Warehouse {i+1} Settings", expanded=True):
        st.subheader(f"Depot Location for Warehouse {i+1}")
        method = st.radio("How to set location?", ["Manual", "Select on Map"], key=f"method_{i}")
        if method == "Manual":
            lat = st.number_input("Latitude", format="%.6f", key=f"lat_{i}")
            lon = st.number_input("Longitude", format="%.6f", key=f"lon_{i}")
        else:
            default_lat = df['latitude'].mean() if not df.empty else 32.5
            default_lon = df['longitude'].mean() if not df.empty else 74.5
            m = folium.Map(location=[default_lat, default_lon], zoom_start=11)
            m.add_child(folium.LatLngPopup())
            md = st_folium(m, width=700, height=350, key=f"map_select_{i}")
            if md and md.get("last_clicked"):
                lat = md["last_clicked"]["lat"]
                lon = md["last_clicked"]["lng"]
                st.success(f"Selected on map: ({lat:.6f}, {lon:.6f})")
            else:
                lat, lon = None, None
                st.info("Click on the map to select a depot location.")

        max_capacity = st.number_input(f"Vehicle Capacity for Warehouse {i+1}", value=270., key=f"capacity_{i}")
        max_shops_per_beat = st.number_input(f"Max Shops per Beat for Warehouse {i+1}", min_value=1, value=50, key=f"shops_{i}")

        warehouse_settings.append({
            "warehouse_num": i+1,
            "lat": lat,
            "lon": lon,
            "capacity": max_capacity,
            "max_shops_per_beat": max_shops_per_beat
        })

# st.write("Warehouse settings debug:", warehouse_settings)

# --------- CACHING SETUP ---------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
config_key_items = []
for wh in warehouse_settings:
    config_key_items.extend([wh['lat'], wh['lon'], wh['capacity'], wh['max_shops_per_beat']])
config_key_items.append(len(warehouse_settings))  # number of warehouses!

shops_assign_cache = os.path.join(
    CACHE_DIR, f"shop_assignment_{file_hash(file_bytes)}_{config_hash(*config_key_items)}.pkl"
)


# ------- 2.5: Assign each shop to the nearest warehouse (CACHED) -------
if os.path.exists(shops_assign_cache):
    with open(shops_assign_cache, "rb") as f:
        df = pickle.load(f)
else:
    valid_warehouses = [
        wh for wh in warehouse_settings
        if wh['lat'] is not None and wh['lon'] is not None
    ]
    if len(valid_warehouses) < len(warehouse_settings):
        st.warning("Please set locations for all warehouses.")
        st.stop()
    depot_coords = [(wh['lat'], wh['lon']) for wh in warehouse_settings]
    def find_nearest_warehouse(row, depot_coords):
        shop_point = (row['latitude'], row['longitude'])
        dists = [geodesic(shop_point, depot).km for depot in depot_coords]
        return int(np.argmin(dists))
    df['warehouse_assignment'] = df.apply(lambda row: find_nearest_warehouse(row, depot_coords), axis=1)
    with open(shops_assign_cache, "wb") as f:
        pickle.dump(df, f)

st.write("Shop assignments to warehouses:")
st.dataframe(df[['shop_id', 'latitude', 'longitude', 'warehouse_assignment']])

# Step 3: Generate Beats Per Warehouse (CACHED!)
all_beats = []
all_beat_orders = []
warehouse_beat_info = []

for i, wh in enumerate(warehouse_settings):
    shops = df[df['warehouse_assignment'] == i].reset_index(drop=True)
    if shops.empty:
        continue

    depot = (wh['lat'], wh['lon'])
    max_capacity = wh['capacity']
    max_shops_per_beat = wh['max_shops_per_beat']

    # --- More robust beat/route cache key ---
    beat_config_items = [depot[0], depot[1], max_capacity, max_shops_per_beat, len(shops), i]
    beat_cache = os.path.join(
        CACHE_DIR,
        f"beats_{file_hash(file_bytes)}_{config_hash(*beat_config_items)}.pkl"
    )
    route_cache = os.path.join(
        CACHE_DIR,
        f"routes_{file_hash(file_bytes)}_{config_hash(*beat_config_items)}.pkl"
    )


    # ---- Beat generation ----
    if os.path.exists(beat_cache) and os.path.exists(route_cache):
        with open(beat_cache, "rb") as f:
            beats, beat_orders = pickle.load(f)
        with open(route_cache, "rb") as f:
            beat_polylines, beat_shop_markers = pickle.load(f)
    else:
        with st.spinner("🚦 Preparing for beat generation..."):
    
            shop_points = list(zip(shops['latitude'], shops['longitude']))
            all_idx = list(range(len(shop_points)))
            unassigned = set(all_idx)
            beats = []
            while unassigned:
                beat = []
                beat_sales = 0
                last_pt = depot
                while unassigned and len(beat) < max_shops_per_beat:
                    nearest_idx = min(unassigned, key=lambda j: geodesic(last_pt, shop_points[j]).km)
                    shop_sale = shops.iloc[nearest_idx]['sales']
                    if beat_sales + shop_sale > max_capacity and beat:
                        break
                    beat.append(nearest_idx)
                    beat_sales += shop_sale
                    last_pt = shop_points[nearest_idx]
                    unassigned.remove(nearest_idx)
                    if len(beat) == max_shops_per_beat:
                        break
                beats.append(beat)

            # ---- TSP for each beat ----
            def build_euclidean_matrix(points):
                n = len(points)
                matrix = np.zeros((n, n))
                for ii in range(n):
                    for jj in range(n):
                        if ii != jj:
                            matrix[ii][jj] = geodesic(points[ii], points[jj]).meters
                return matrix

            def solve_tsp(dist_matrix):
                n = len(dist_matrix)
                manager = pywrapcp.RoutingIndexManager(n, 1, 0)
                routing = pywrapcp.RoutingModel(manager)
                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return int(dist_matrix[from_node][to_node])
                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                search_params = pywrapcp.DefaultRoutingSearchParameters()
                search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                solution = routing.SolveWithParameters(search_params)
                if not solution:
                    return list(range(n))
                idx = routing.Start(0)
                route = []
                while not routing.IsEnd(idx):
                    route.append(manager.IndexToNode(idx))
                    idx = solution.Value(routing.NextVar(idx))
                route.append(manager.IndexToNode(idx))
                return route

            beat_orders = []
            beat_polylines = []
            beat_shop_markers = []
            MAX_ORS_WAYPOINTS = 50
            colors = []
            palette = cm.get_cmap('tab20', len(beats))
            color_list = [
                '#%02x%02x%02x' % tuple(int(255 * x) for x in palette(j)[:3])
                for j in range(len(beats))
            ]

            with st.spinner(f"Processing beats and routes for Warehouse {i+1}..."):
                for beat_num, beat in enumerate(beats):
                    if not beat:
                        continue
                    beat_points = [depot] + [shop_points[j] for j in beat]
                    dist_matrix = build_euclidean_matrix(beat_points)
                    tsp_order = solve_tsp(dist_matrix)
                    beat_shop_order = [beat[j-1] for j in tsp_order[1:-1]]
                    beat_orders.append(beat_shop_order)

                    # ORS road polyline per beat
                    color = color_list[beat_num % len(color_list)]
                    points = [depot] + [shop_points[j] for j in beat_shop_order] + [depot]
                    subroutes = []
                    n = len(points)
                    if n <= MAX_ORS_WAYPOINTS:
                        subroutes.append(points)
                    else:
                        for start in range(0, n-1, MAX_ORS_WAYPOINTS-1):
                            leg = points[start: start + MAX_ORS_WAYPOINTS]
                            if leg[-1] != points[-1]:
                                leg.append(points[-1])
                            subroutes.append(leg)
                    beat_polyline_points = []
                    for subroute in subroutes:
                        coords = [[lon, lat] for lat, lon in subroute]
                        try:
                            directions = client.directions(
                                coordinates=coords,
                                profile='driving-car',
                                format='geojson'
                            )
                            for pt in directions['features'][0]['geometry']['coordinates']:
                                beat_polyline_points.append({"lat": pt[1], "lng": pt[0]})
                        except Exception as e:
                            for lat, lon in subroute:
                                beat_polyline_points.append({"lat": lat, "lng": lon})
                    shops_in_this_beat = []
                    for order, shop_idx in enumerate(beat_shop_order, 1):
                        shop = shops.iloc[shop_idx]
                        shops_in_this_beat.append({
                            "lat": shop["latitude"],
                            "lng": shop["longitude"],
                            "shop_id": shop["shop_id"],
                            "order": order,
                            "sales": shop["sales"],
                            "beat_num": beat_num + 1,
                            "color": color,
                            "total_shops": len(beat_shop_order),
                            "warehouse": i + 1
                        })
                    beat_polylines.append({
                        "color": color,
                        "points": beat_polyline_points,
                        "warehouse": i + 1
                    })
                    beat_shop_markers.append(shops_in_this_beat)

            with open(beat_cache, "wb") as f:
                pickle.dump((beats, beat_orders), f)
            with open(route_cache, "wb") as f:
                pickle.dump((beat_polylines, beat_shop_markers), f)

        all_beats.append(beats)
        all_beat_orders.append(beat_orders)
        
    warehouse_beat_info.append({
        "depot": depot,
        "beats": beats,
        "beat_orders": beat_orders,
        "shops": shops,
        "beat_polylines": beat_polylines,
        "beat_shop_markers": beat_shop_markers
    })

st.success(f"✅ Beats and road routes (cached) for all warehouses!")
    
# st.write(f"Generated {len(beats)} beats for warehouse {i+1}")

st.subheader("Number of beats per warehouse")
for idx, wh in enumerate(warehouse_beat_info):
    st.write(f"Warehouse {idx+1}: {len(wh['beats'])} beats")

if not warehouse_beat_info:
    st.error("No beats/routes were generated for any warehouse! Please check your uploaded shop data, warehouse locations, and constraints. Make sure every warehouse has at least one shop assigned and that depot locations are set.")
    st.stop()
    
st.header("4️⃣ Visualize All Optimized Beats & Routes (Real Road Paths)")

# 1. Select warehouse
warehouse_options = [f"Warehouse {i+1}" for i in range(len(warehouse_beat_info))]
selected_wh_idx = st.selectbox("Select Warehouse", options=range(len(warehouse_beat_info)), format_func=lambda x: warehouse_options[x])

selected_wh = warehouse_beat_info[selected_wh_idx]
beat_polylines = selected_wh["beat_polylines"]
beat_shop_markers = selected_wh["beat_shop_markers"]
depot = selected_wh["depot"]

if not warehouse_beat_info:
    st.error("No warehouses with beats found! Please make sure you uploaded valid shop data, set ALL warehouse depot locations, and assigned shops to warehouses.")
    st.stop()
    
# 2. Checkboxes for beats of selected warehouse
st.subheader(f"👁️ Select Beats to Display for {warehouse_options[selected_wh_idx]}")
checked_indices = []
for i in range(len(beat_polylines)):
    checked = st.checkbox(f"Show Beat {i+1}", value=True, key=f"show_beat_{selected_wh_idx}_{i+1}")
    if checked:
        checked_indices.append(i)
if not checked_indices:
    st.warning("Select at least one beat to display on the map.")
    st.stop()

filtered_beat_polylines = [beat_polylines[i] for i in checked_indices]
filtered_beat_shop_markers = [beat_shop_markers[i] for i in checked_indices]
shop_markers_json = json.dumps(filtered_beat_shop_markers, default=convert_np)
polylines_json = json.dumps(filtered_beat_polylines, default=convert_np)

depot_marker = {
    "lat": depot[0],
    "lng": depot[1],
    "title": f"Depot ({warehouse_options[selected_wh_idx]})"
}
depot_json = json.dumps(depot_marker, default=convert_np)

# ... rest of your mapbox visualization code
html_code = f"""
<div id="map" style="height:700px;width:100%;"></div>
<link href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet" />
<script src="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>
<script>
    var polylines = {polylines_json};
    mapboxgl.accessToken = '{MAPBOX_TOKEN}';
    var map = new mapboxgl.Map({{
        container: 'map',
        style: 'mapbox://styles/mapbox/streets-v12',
        center: [{depot[1]}, {depot[0]}],
        zoom: 12
    }});

    // Depot marker
    var depotMarker = new mapboxgl.Marker({{
        color: "blue"
    }}).setLngLat([{depot[1]}, {depot[0]}]).setPopup(
        new mapboxgl.Popup().setHTML('<b>Depot</b>')
    ).addTo(map);

    // --- Draw all polylines and layers INSIDE ONE 'load' event ---
    map.on('load', function() {{
        for (var i = 0; i < polylines.length; i++) {{
            var points = polylines[i].points.map(pt => [pt.lng, pt.lat]);
            map.addSource('route' + i, {{
                'type': 'geojson',
                'data': {{
                    'type': 'Feature',
                    'geometry': {{
                        'type': 'LineString',
                        'coordinates': points
                    }}
                }}
            }});
            map.addLayer({{
                'id': 'route' + i,
                'type': 'line',
                'source': 'route' + i,
                'layout': {{}},
                'paint': {{
                    'line-color': polylines[i].color,
                    'line-width': 4
                }}
            }});
        }}
    }});

    // Draw shop "dots" with colored popup per beat
    var shop_markers = {shop_markers_json};
    for (var b = 0; b < shop_markers.length; b++) {{
        for (var j = 0; j < shop_markers[b].length; j++) {{
            var shop = shop_markers[b][j];
            var el = document.createElement('div');
            el.className = 'marker';
            el.style.background = shop.color;
            el.style.width = '14px';
            el.style.height = '14px';
            el.style.borderRadius = '50%';
            el.style.border = '2px solid white';
            new mapboxgl.Marker(el)
                .setLngLat([shop.lng, shop.lat])
                .setPopup(
                    new mapboxgl.Popup().setHTML(
                        `<b>Beat #:</b> ${{shop.beat_num}}<br>
                        <b>Shop:</b> ${{shop.shop_id}}<br>
                        <b>Shop Order in Beat:</b> ${{shop.order}}<br>
                        <b>Total Shops in Beat:</b> ${{shop.total_shops}}<br>
                        <b>Sales:</b> ${{shop.sales}}`
                    )
                )
                .addTo(map);
        }}
    }}
</script>
"""
components.html(html_code, height=700, width=1100)

# Optional: Download beat assignments
output_data = []
for wh_idx, wh in enumerate(warehouse_beat_info, 1):
    for beat_num, (beat, shop_order) in enumerate(zip(wh["beats"], wh["beat_orders"]), 1):
        for order, shop_idx in enumerate(shop_order, 1):
            shop = wh["shops"].iloc[shop_idx]
            output_data.append({
                "Warehouse": wh_idx,
                "Beat": beat_num,
                "Visit Order": order,
                "Shop ID": shop['shop_id'],
                "Latitude": shop['latitude'],
                "Longitude": shop['longitude'],
                "Sales": shop['sales'],
            })
out_df = pd.DataFrame(output_data)
st.download_button("Download Optimized Beat Assignments", out_df.to_csv(index=False), file_name="optimized_beats.csv")



