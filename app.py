import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import folium
from streamlit_folium import st_folium
import time
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import numpy as np

#both min max:  loop until min max fine

st.set_page_config(page_title="Smart Beat Optimizer", layout="wide")
st.title("🚚 Beat Planning & Route Optimization (HDBSCAN, Min/Max Shops per Beat)")

# 1. Upload Shop Data
st.header("1️⃣ Upload Shop Data")
uploaded = st.file_uploader("Upload CSV (columns: shop_id, latitude, longitude, address, sales)", type=["csv"])
if not uploaded:
    st.info("Upload your CSV to proceed.")
    st.stop()

df = pd.read_csv(uploaded)
df['latitude'].replace(0, np.nan, inplace=True)
df['longitude'].replace(0, np.nan, inplace=True)
lat_mean = df['latitude'].mean()
lon_mean = df['longitude'].mean()
df['latitude'].fillna(lat_mean, inplace=True)
df['longitude'].fillna(lon_mean, inplace=True)

# 2. Depot (Warehouse) Location
st.header("2️⃣ Set Depot (Warehouse Location)")
with st.expander("Depot Settings", expanded=True):
    lat = st.number_input("Depot Latitude", value=float(df['latitude'].mean()))
    lon = st.number_input("Depot Longitude", value=float(df['longitude'].mean()))
    depot = (lat, lon)

# 3. Min/Max Shops Per Beat
st.header("3️⃣ Beat Clustering (HDBSCAN + Max Shop Split)")
min_cluster_size = st.slider("Minimum shops per beat", min_value=2, max_value=50, value=5)
max_cluster_size = st.slider("Maximum shops per beat", min_value=min_cluster_size, max_value=100, value=20)

# 4. Clustering with HDBSCAN, then split large clusters

def cluster_shops_strict_minmax(df, hdbscan_labels, min_shops, max_shops):
    df = df.copy()
    df['beat_id'] = hdbscan_labels

    # Assign outliers (-1) to nearest cluster centroid
    outlier_mask = df['beat_id'] == -1
    if outlier_mask.any():
        valid_clusters = df[~outlier_mask]['beat_id'].unique()
        centroids = {label: df[df['beat_id']==label][['latitude','longitude']].mean() for label in valid_clusters}
        for idx, row in df[outlier_mask].iterrows():
            dists = {label: geodesic((row['latitude'], row['longitude']), tuple(centroids[label])).km for label in centroids}
            nearest_label = min(dists, key=dists.get)
            df.at[idx, 'beat_id'] = nearest_label

    # --- Iteratively split large clusters ---
    while True:
        counts = df['beat_id'].value_counts()
        too_big = counts[counts > max_shops]
        if too_big.empty:
            break
        for label in too_big.index:
            shops = df[df['beat_id'] == label]
            n_split = int(np.ceil(len(shops) / max_shops))
            coords = shops[['latitude', 'longitude']].to_numpy()
            if len(shops) < 2:
                continue
            km = KMeans(n_clusters=n_split, n_init=10, random_state=42)
            split_labels = km.fit_predict(coords)
            for sub in range(n_split):
                new_label = f"{label}_SPLIT{sub}_{np.random.randint(1e9)}"
                df.loc[shops.index[split_labels == sub], 'beat_id'] = new_label

    # --- Iteratively merge small clusters ---
    while True:
        counts = df['beat_id'].value_counts()
        too_small = counts[counts < min_shops]
        if too_small.empty:
            break
        for label in too_small.index:
            shops = df[df['beat_id'] == label]
            all_other = df[df['beat_id'] != label]
            if all_other.empty:
                continue
            centroids = df.groupby('beat_id')[['latitude', 'longitude']].mean()
            c0 = centroids.loc[label]
            min_dist = float('inf')
            best_label = None
            for other in centroids.index:
                if other == label: continue
                c1 = centroids.loc[other]
                d = geodesic(tuple(c0), tuple(c1)).km
                if d < min_dist:
                    min_dist = d
                    best_label = other
            df.loc[shops.index, 'beat_id'] = best_label

    # Re-label beats as B1, B2, ...
    unique_beats = df['beat_id'].unique()
    beat_id_map = {label: f"B{i+1}" for i, label in enumerate(unique_beats)}
    df['beat_id'] = df['beat_id'].map(beat_id_map)
    return df

# 5. TSP Routing per Beat
def solve_tsp_for_beat(beat_df, depot):
    locs = [depot] + list(zip(beat_df.latitude, beat_df.longitude))
    N = len(locs)
    dist = [[int(geodesic(locs[i], locs[j]).km * 1000) for j in range(N)] for i in range(N)]
    mgr = pywrapcp.RoutingIndexManager(N, 1, 0)
    model = pywrapcp.RoutingModel(mgr)
    def cost(i, j): return dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)]
    cid = model.RegisterTransitCallback(cost)
    model.SetArcCostEvaluatorOfAllVehicles(cid)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 20
    sol = model.SolveWithParameters(params)
    if not sol:
        order = list(range(len(beat_df)))
    else:
        idx = model.Start(0)
        order = []
        while not model.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node > 0:
                order.append(node - 1)
            idx = sol.Value(model.NextVar(idx))
    routed = beat_df.reset_index(drop=True).iloc[order].copy()
    routed['route_order'] = range(1, len(routed) + 1)
    return routed

def relabel_beats(df):
    unique_beats = sorted(df['beat_id'].unique())
    beat_map = {beat: f"B{idx+1}" for idx, beat in enumerate(unique_beats)}
    df['beat_id'] = df['beat_id'].map(beat_map)
    return df

# 6. Main Logic: Cluster & Route
with st.spinner("Clustering and optimizing routes..."):
    t0 = time.time()
    coords_rad = np.radians(df[['latitude', 'longitude']].to_numpy())
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=1,metric='haversine')
    hdbscan_labels = clusterer.fit_predict(coords_rad)
   
    clustered = cluster_shops_strict_minmax(df, hdbscan_labels, min_cluster_size, max_cluster_size)
    
    clustered = relabel_beats(clustered)
    t1 = time.time()
    st.info(f"Clustering completed in {t1 - t0:.2f} seconds.")
    beats = list(clustered['beat_id'].unique())
    n_beats = len(beats)
    results = []
    tsp_start = time.time()
    progress_bar = st.progress(0, text="Optimizing TSP routes for all beats...")
    
    for i, beat in enumerate(beats):
        beat_df = clustered[clustered['beat_id'] == beat].copy()
        t_beat = time.time()
        routed = solve_tsp_for_beat(beat_df, depot)
        t_beat_elapsed = time.time() - t_beat
        st.write(f"TSP for {beat} ({len(beat_df)} shops) done in {t_beat_elapsed:.2f} sec")
        routed['beat_id'] = beat
        results.append(routed)
        progress_bar.progress((i + 1) / n_beats, text=f"Optimized {i + 1} / {n_beats} beats")
    progress_bar.empty()
    tsp_total = time.time() - tsp_start
    st.info(f"All TSP routes computed in {tsp_total:.2f} seconds.")
    final_all = pd.concat(results, ignore_index=True)

st.success("Beats and routes computed!")
st.write("Sample of results:", final_all[['shop_id','beat_id','route_order']].head(20))

# 7. Visualization
st.header("4️⃣ Visualize Beats on Map")
beat_list = sorted(final_all['beat_id'].unique())
selected_beats = st.multiselect("Select Beats to Display", beat_list, default=beat_list[:3])
color_palette = ['blue', 'red', 'purple', 'orange', 'green', 'pink', 'cadetblue', 'lightgray', 'black', 'gold']

if selected_beats:
    m = folium.Map(location=depot, zoom_start=11)
    for idx, beat in enumerate(selected_beats):
        sub = final_all[final_all['beat_id'] == beat].sort_values('route_order')
        pts = list(zip(sub.latitude, sub.longitude))
        color = color_palette[idx % len(color_palette)]
        if pts:
            folium.PolyLine([depot, pts[0]], color='darkgreen', weight=3, dash_array='5,10').add_to(m)
            folium.PolyLine(pts, color=color, weight=4, tooltip=beat).add_to(m)
            folium.PolyLine([pts[-1], depot], color='darkgreen', weight=3, dash_array='5,10').add_to(m)
            for i, r in sub.iterrows():
                folium.CircleMarker(
                    location=(r.latitude, r.longitude),
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"{beat} (#{r.route_order})"
                ).add_to(m)
    st_folium(m, width=900, height=600)
else:
    st.info("Select at least one beat to view.")

# 8. Download results
st.download_button("Download Full Results CSV", final_all.to_csv(index=False), file_name="clustered_routed_output.csv")
