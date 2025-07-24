import streamlit as st
import pandas as pd
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import time

#manual and auto both.... final
#latest-updated
#multi warehouses -  0,0 handled
#final

# ─── Helper to pick the best k beats by minimizing max inter‑beat distance ─────────────────
def find_nearest_beats(centroids: pd.DataFrame, k: int):
    pts = list(centroids[['latitude','longitude']].itertuples(index=False, name=None))
    beat_ids = centroids['beat_id'].tolist()
    n = len(pts)
    if k >= n:
        return beat_ids

    # build distance matrix
    dist = [[geodesic(pts[i], pts[j]).km for j in range(n)] for i in range(n)]

    best_group = None
    best_max = float('inf')
    for i in range(n):
        # sort other beats by distance from beat i
        neighbors = sorted(range(n), key=lambda j: dist[i][j])[:k]
        # compute max pairwise distance within this group
        max_d = max(dist[a][b] for a in neighbors for b in neighbors)
        if max_d < best_max:
            best_max, best_group = max_d, neighbors

    return [beat_ids[idx] for idx in best_group]


# ─── Cached Helpers ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False) 
def compute_routes(df, depot, min_shops, max_shops, auto_beats=False):
    """
    1) Fill missing lat/lon (including 0,0), 2) Cluster shops into "beats," 
    3) Solve TSP per beat, 4) Return ordered routes.
    """
    # ◉ TREAT ZEROS AS MISSING
    zero_mask = (df['latitude'] == 0) & (df['longitude'] == 0)
    df.loc[zero_mask, ['latitude', 'longitude']] = pd.NA

    # Geocoding missing latitude/longitude
    geolocator = Nominatim(user_agent="beat_optimizer_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    missing = df['latitude'].isna() | df['longitude'].isna()
    for i, row in df[missing].iterrows():
        try:
            location = geocode(row['address'])
            if location:
                df.at[i, 'latitude'] = location.latitude
                df.at[i, 'longitude'] = location.longitude
        except Exception:
            pass

    # Fill any remaining missing values with global mean
    lat_mean = df['latitude'].mean()
    lon_mean = df['longitude'].mean()
    df['latitude'].fillna(lat_mean, inplace=True)
    df['longitude'].fillna(lon_mean, inplace=True)

    # … rest of compute_routes unchanged …

    df['longitude'].fillna(lon_mean, inplace=True)
    
   
    # Clustering into beats via angle-sweep
    df_clust = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

    # angle sweep (always needed for geography)
    df_clust['angle'] = df_clust.apply(
        lambda r: math.atan2(r['latitude'] - depot[0],
                             r['longitude'] - depot[1]),
        axis=1
    )
    df_sorted = df_clust.sort_values('angle').reset_index(drop=True)

    if auto_beats:
        # fixed‐size chunks of 50 shops
        beat_size = 50
        S = len(df_sorted)
        full = S // beat_size
        rem  = S % beat_size
        sizes = [beat_size]*full + ([rem] if rem else [])
    else:
        # your existing min/max logic
        S, m, M = len(df_sorted), min_shops, max_shops
        k, r = S // M, S - (S//M)*M
        sizes = []
        if k == 0:
            sizes = [S]
        else:
            if r >= m:
                sizes = [M]*k + [r]
            else:
                need = m - r
                per, rem2 = divmod(need, k)
                for j in range(k):
                    loss = per + (1 if j < rem2 else 0)
                    sizes.append(M - loss)
                sizes.append(m)

    # now slice into beats exactly as before
    beats, idx = [], 0
    for bi, sz in enumerate(sizes, start=1):
        chunk = df_sorted.iloc[idx: idx + sz].copy()
        chunk['beat_id'] = f"B{bi}"
        beats.append(chunk)
        idx += sz


    # Solve TSP for each beat
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    optimized = []
    for beat in beats:
        locs = [depot] + list(zip(beat.latitude, beat.longitude))
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
            order = list(range(len(beat)))
        else:
            idx = model.Start(0)
            order = []
            while not model.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node > 0:
                    order.append(node - 1)
                idx = sol.Value(model.NextVar(idx))

        routed = beat.reset_index(drop=True).iloc[order].copy()
        routed['route_order'] = range(1, len(routed) + 1)
        routed['beat_id'] = beat.beat_id.iloc[0]
        optimized.append(routed)

    final_df = pd.concat(optimized, ignore_index=True)
    return final_df


@st.cache_data(show_spinner=False)
def build_map_for_beat(final_df, depot, beat_id):
    """Draws a Folium map for a single beat sequence."""
    view = final_df[final_df.beat_id == beat_id].sort_values('route_order')
    m = folium.Map(location=[view.latitude.mean(), view.longitude.mean()], zoom_start=13)
    pts = list(zip(view.latitude, view.longitude))

    folium.PolyLine([depot, pts[0]], color='green',dash_array='5, 10', weight=4).add_to(m)
    folium.PolyLine(pts, color='blue', weight=3).add_to(m)
    folium.PolyLine([pts[-1], depot], color='green', dash_array='5, 10', weight=4).add_to(m)

    for _, r in view.iterrows():
        folium.Marker(
            location=(r.latitude, r.longitude),
            icon=folium.DivIcon(html=f"<div style='font-size:12px;color:blue'>{r.route_order}</div>")
        ).add_to(m)
    return m


# ─── Main App ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Beat & Route Optimizer", layout="wide")
st.title("🚚 Beat Planning & Route Optimization — Multi‑Warehouse")

# Step 1: Warehouses & Beat Size
st.header("🔧 Step 1: Warehouses & Beat Size")
num_wh = st.number_input("Number of Warehouses", min_value=1, max_value=5, value=1, step=1)
warehouses = []
for i in range(int(num_wh)):
    with st.expander(f"🏭 Warehouse {i+1} Settings", expanded=True):
        method = st.radio("Depot location:", ["Manual", "Click on Map"], key=f"method_{i}")
        if method == "Manual":
            lat = st.number_input("Latitude", format="%.6f", key=f"lat_{i}")
            lon = st.number_input("Longitude", format="%.6f", key=f"lon_{i}")
        else:
            st.write("Click on the map to set the depot:")
            base_map = folium.Map(location=[32.5, 74.5], zoom_start=12)
            base_map.add_child(folium.LatLngPopup())
            md = st_folium(base_map, width=700, height=300, key=f"map_select_{i}")
            if md and md.get("last_clicked"):
                lat = md["last_clicked"]["lat"]
                lon = md["last_clicked"]["lng"]
            else:
                lat, lon = None, None

        
        mode = st.radio(
            "Beat Generation:", ["Manual", "Auto-suggest"], key=f"beat_mode_{i}")
        auto_beats = (mode == "Auto-suggest")

        if mode == "Manual":
            min_shops = st.number_input(
                "Min shops per beat", min_value=1, value=50, key=f"min_shops_{i}")
            max_shops = st.number_input(
                "Max shops per beat", min_value=min_shops, value=70, key=f"max_shops_{i}")
        else:
            # Auto-suggest path: disable manual inputs
            min_shops = None
            max_shops = None
        
        salesmen = st.number_input("Number of Salesmen (vehicles)", min_value=1, value=5, key=f"salesmen_{i}")
        
        # min_shops = st.number_input("Min shops/beat", min_value=1, value=50, key=f"min_shops_{i}")
        # max_shops = st.number_input("Max shops/beat", min_value=min_shops, value=70, key=f"max_shops_{i}")
        
        # — Dynamic Cars —
        num_cars = st.number_input("Number of Cars", min_value=0, value=2, key=f"num_cars_{i}")
        cars = []
        for j in range(int(num_cars)):
            with st.expander(f"🚗 Car {j+1} Details", expanded=False):
                name = st.text_input("Car Name", value=f"car{j+1}", key=f"car_name_{i}_{j}")
                cap  = st.number_input("Car Carry Capacity (packages)", min_value=1, value=100, key=f"car_cap_{i}_{j}")
                beats_to_cover = st.number_input("Beats to cover", min_value=1, value=1, key=f"car_beats_{i}_{j}")
                cars.append({"name":name, "capacity":cap, "beat_cover":beats_to_cover})

        # — Dynamic Bikes —
        num_bikes = st.number_input("Number of Bikes", min_value=0, value=3, key=f"num_bikes_{i}")
        bikes = []
        for j in range(int(num_bikes)):
            with st.expander(f"🏍️ Bike {j+1} Details", expanded=False):
                name = st.text_input("Bike Name", value=f"bike{j+1}", key=f"bike_name_{i}_{j}")
                cap  = st.number_input("Bike Carry Capacity (packages)", min_value=1, value=50, key=f"bike_cap_{i}_{j}")
                beats_to_cover = st.number_input("Beats to cover", min_value=1, value=1, key=f"bike_beats_{i}_{j}")
                bikes.append({"name":name, "capacity":cap, "beat_cover":beats_to_cover})
        
        warehouses.append({
            'depot': (lat, lon),
            'min_shops': min_shops,
            'max_shops': max_shops,
            'cars':        cars,
            'bikes':       bikes,
            'auto_beats': auto_beats
        })

# Warning if any depot not set
def incomplete_depot(w): return w['depot'][0] is None or w['depot'][1] is None
if any(incomplete_depot(w) for w in warehouses):
    st.warning("Please set all depot locations above before uploading your shop file.")

# Step 2: Upload & Compute
st.header("📂 Step 2: Upload & Compute")
uploaded = st.file_uploader("CSV (shop_id, shop_name, latitude, longitude, address, sales)", type=["csv", "xlsx"])
final_all = None
if uploaded and not any(incomplete_depot(w) for w in warehouses):
    df = pd.read_csv(uploaded)
    st.success("✅ File loaded")

    # ◉ TREAT ZEROS AS MISSING HERE TOO
    zero_mask_main = (df['latitude'] == 0) & (df['longitude'] == 0)
    df.loc[zero_mask_main, ['latitude', 'longitude']] = pd.NA

    # Fill missing coords for assignment
    lat_mean = df['latitude'].mean()
    lon_mean = df['longitude'].mean()
    df['latitude'].fillna(lat_mean, inplace=True)
    df['longitude'].fillna(lon_mean, inplace=True)

    st.success("✅ Missing lat/lon values (including 0,0) handled successfully.")
    # … rest of your main logic unchanged …

    # Assign each shop to nearest warehouse
    for idx, wh in enumerate(warehouses, start=1):
        df[f'dist_wh_{idx}'] = df.apply(
            lambda r: geodesic((r.latitude, r.longitude), wh['depot']).km, axis=1
        )
    dist_cols = [f'dist_wh_{i}' for i in range(1, len(warehouses)+1)]
    df['warehouse_id'] = df[dist_cols].idxmin(axis=1).apply(lambda s: int(s.split('_')[-1]))

 # Display total shops per warehouse
    for idx in range(1, len(warehouses) + 1):
        count = df[df['warehouse_id'] == idx].shape[0]
        st.write(f"🏭 For Warehouse {idx}: Total shops are {count}")
        
        
    # Show loading spinner while routes are being computed
    with st.spinner("Optimizing routes, please wait..."):
        results = []
        for wid, wh in enumerate(warehouses, start=1):
            subset = df[df.warehouse_id == wid].copy()
            if not subset.empty:
                routed = compute_routes(
                subset,
                wh['depot'],
                wh.get('min_shops'),
                wh.get('max_shops'),
                auto_beats=wh.get('auto_beats', False)
            )
                routed['warehouse_id'] = wid
                results.append(routed)
        final_all = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        st.success("✅ Route optimization complete!")
        st.dataframe(final_all[['shop_id','shop_name','warehouse_id','beat_id','route_order']])

    # final summary: beats & counts
        for i in range(1,num_wh+1):
            wh_df = final_all[final_all.warehouse_id==i]
            beats = wh_df.beat_id.unique()
            st.write(f"Warehouse {i}: {len(beats)} beats")
            for b in beats:
                cnt = wh_df[wh_df.beat_id==b].shape[0]
                st.write(f"• {b}: {cnt} shops")
                
        
        # After final_all has been built, before visualization:
    # ─── Step 3: Vehicle Assignments per Warehouse (Sales‑based) ────────────────
        if final_all is not None and not final_all.empty:
            st.header("🚚 Vehicle Assignments per Warehouse (Shops + Sales)")

            for wid, wh in enumerate(warehouses, start=1):
                st.subheader(f"Warehouse {wid}")

                # 1) Compute total sales per beat
                beat_stats = (
                    final_all[final_all.warehouse_id == wid]
                    .groupby('beat_id')
                    .agg(total_sales=('sales','sum'))
                    .reset_index()
                )

                # 2) Compute centroids for grouping
                centroids = (
                    final_all[final_all.warehouse_id == wid]
                    .groupby('beat_id')
                    .agg(
                        latitude = ('latitude','mean'),
                        longitude= ('longitude','mean')
                    )
                    .reset_index()
                )

                # Merge so beat_stats has lat/lon too
                beat_stats = beat_stats.merge(centroids, on='beat_id')
                
                #extra
                
                all_vehicles = [(v, 'Car')  for v in wh['cars']] \
                            + [(v, 'Bike') for v in wh['bikes']]
                max_salesmen  = st.session_state[f"salesmen_{wid-1}"]
                vehicles_to_run = all_vehicles[:max_salesmen]
                
                # 3) Start with all beats unassigned
                remaining_beats = set(beat_stats['beat_id'])
                assignments = []
                combined_warnings = []

                # 4) For each vehicle (cars first, then bikes)…
                
                for veh, label in vehicles_to_run:
                    # find k & sales exactly like before
                    k = veh['beat_cover']
                    pool = beat_stats[beat_stats.beat_id.isin(remaining_beats)].reset_index(drop=True)
                    chosen_beats = find_nearest_beats(
                        pool[['beat_id','latitude','longitude']], k
                    )
                    sales_assigned = int(pool[pool.beat_id.isin(chosen_beats)]['total_sales'].sum())
                    fulfilled = (sales_assigned <= veh['capacity'])

                    assignments.append({
                        'Vehicle':        veh['name'],
                        'Type':           label,
                        'Beats assigned': ", ".join(chosen_beats) if fulfilled else "—",
                        'Sales assigned': sales_assigned   if fulfilled else 0,
                        'Capacity':       veh['capacity'],
                        'Fulfilled':      fulfilled
                    })

                    if fulfilled:
                        remaining_beats -= set(chosen_beats)
                    else:
                        combined_warnings.append(
                            f"• **{veh['name']}** ({label}) is overloaded: "
                            f"{sales_assigned} > {veh['capacity']}.  "
                            "Their beats remain unassigned."
                        )
                        
                # 5) Display the assignment table
                df_assign = pd.DataFrame(assignments)
                # st.table(df_assign)
                # 4) Build DataFrame (no SalesmanID yet)
            

                # 5) Compute SalesmanID only for rows with assigned beats (i.e. Beats assigned != "—")
                salesman_ids = []
                counter = 1
                for _, row in df_assign.iterrows():
                    if row['Beats assigned'] != "—":     # or use row['Fulfilled'] == True
                        salesman_ids.append(f"S{counter}")
                        counter += 1
                    else:
                        salesman_ids.append("")           # leave blank for bike2

                df_assign.insert(0, 'SalesmanID', salesman_ids)
                
        
                st.subheader("🧑‍💼 Salesman ⇄ Vehicle ⇄ Beat Assignments")
                st.table(df_assign)
                
                 # ─── 5) Single combined warning ───────────────────────────────────────
                if combined_warnings or remaining_beats:
                    msgs = []
                    if combined_warnings:
                        msgs.append("⚠️ **Assignment Warnings:**\n" + "\n".join(combined_warnings))
                    if remaining_beats:
                        msgs.append(
                            "❗️ **Unassigned Beats:** "
                            + ", ".join(sorted(remaining_beats))
                            + ".  Please increase salesmen or adjust beats‑to‑cover option."
                        )
                    st.warning("\n\n".join(msgs))

                # ─── 6) (Optional) Show unassigned‑beats table with sales ─────────────
                if remaining_beats:
                    df_un = (
                        beat_stats[beat_stats.beat_id.isin(remaining_beats)]
                        .rename(columns={'beat_id':'Beat ID','total_sales':'Total Sales'})
                        [['Beat ID','Total Sales']]
                    )
                    st.subheader("❗️ Unassigned Beats")
                    st.table(df_un)
            #     for _, row in df_assign[df_assign['Fulfilled'] == False].iterrows():
            #         st.warning(
            #     f"⚠️ {row['Vehicle']} (Salesman {row['SalesmanID']}) "
            #     f"is overloaded: assigned {row['Sales assigned']} > capacity {row['Capacity']}. "
            #     "Please increase its carrying capacity."
            # )
                    
                    # ─── Step 4: Salesman‑Beat Route Visualization ─────────────────────────────
            st.subheader(f"🗺️ Warehouse {wid} · Salesman Beat Routes")

            # 1) pick a salesman
            sal_list = df_assign['SalesmanID'].tolist()
            sel_sal = st.selectbox("Select Salesman", sal_list, key=f"sal_{wid}")

            # 2) parse their beats
            beats_str = df_assign.loc[
                df_assign['SalesmanID'] == sel_sal, 'Beats assigned'
            ].iloc[0]
            beat_list = [b.strip() for b in beats_str.split(',') if b.strip()]

            # 3) gather all points for centering
            final_wh = final_all[final_all.warehouse_id == wid]
            coords = []
            for beat in beat_list:
                pts = final_wh[final_wh.beat_id == beat][['latitude','longitude']].values.tolist()
                coords += pts

            # 4) initialize map
            if coords:
                center = [sum(p[0] for p in coords)/len(coords),
                        sum(p[1] for p in coords)/len(coords)]
            else:
                center = wh['depot']
            m_sal = folium.Map(location=center, zoom_start=12)

            # 5) draw each beat in its own color
            colors = ['blue','red','purple','orange','darkred','lightblue','cadetblue']
            for idx, beat in enumerate(beat_list):
                dfb = final_wh[final_wh.beat_id == beat].sort_values('route_order')
                pts = list(zip(dfb.latitude, dfb.longitude))
                color = colors[idx % len(colors)]
                # folium.PolyLine(pts, color=color, weight=4, tooltip=beat).add_to(m_sal)
                # 1) depot → first shop
                folium.PolyLine(
                    [wh['depot'], pts[0]],
                    color='green',
                    weight=3,
                    dash_array='5, 10',
                    tooltip=f"{beat} start"
                ).add_to(m_sal)

                # 2) shop‑to‑shop route
                folium.PolyLine(
                    pts,
                    color=color,
                    weight=4,
                    tooltip=beat
                ).add_to(m_sal)

                # 3) last shop → depot
                folium.PolyLine(
                    [pts[-1], wh['depot']],
                    color='green',
                    weight=3,
                    dash_array='5, 10',
                    tooltip=f"{beat} return"
                ).add_to(m_sal)

                for _, r in dfb.iterrows():
                    folium.CircleMarker(
                        location=(r.latitude, r.longitude),
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"{beat} (#{r.route_order})"
                    ).add_to(m_sal)

            # 6) render
            st_folium(m_sal, width=800, height=600, key=f"map_sal_{wid}")
                

        # Step 5: Visualize per Warehouse

            st.header("🗺️ Step 3: Visualize Beat per Warehouse")
            wh_choice = st.selectbox("Select Warehouse", sorted(final_all.warehouse_id.unique()), key="wh_choice")
            beat_list = sorted(final_all[final_all.warehouse_id == wh_choice].beat_id.unique())
            beat_choice = st.selectbox("Select Beat", beat_list, key="beat_view")


#edited
            sub_final = final_all[(final_all.warehouse_id == wh_choice) & (final_all.beat_id == beat_choice)]
            depot_pt = warehouses[wh_choice-1]['depot']
            map_obj = build_map_for_beat(sub_final, depot_pt, beat_choice)
            st_folium(map_obj, width=800, height=600, key="map_view")
            
            
