import streamlit as st
import pandas as pd
import numpy as np
import gdown
import joblib
import os
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib.patches import Patch


import streamlit as st

# CSS para personalizar tablas
st.markdown("""
    <style>
    /* Cambiar encabezado de tablas */
    thead tr th {
        color: #FFD700 !important;   /* dorado */
        font-weight: bold;
        font-size: 16px;
    }

    /* Cambiar color de las celdas */
    tbody tr td {
        color: #FFFFFF !important;   /* blanco */
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------
# Sidebar Logo
# ----------------------------
st.sidebar.image("Logo.jpg", width=110)

# ----------------------------
# Load model, scalers and preprocessor
# ----------------------------
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

MODEL_FILE = "final_rf_tuned_fast_model.pkl"
GOOGLE_DRIVE_ID = "1Fmw782ET3fxqZphucD-PKrLFjFa6Xccq"
URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

if not os.path.exists(MODEL_FILE):
    print("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_FILE, quiet=False)

rf = joblib.load(MODEL_FILE)
#rf = joblib.load("final_rf_tuned_fast_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
output_columns = joblib.load("output_columns.pkl")

# Historical dataset for comparisons
historical_csv = "synthetic_full_dataset.csv"
df_hist = pd.read_csv(historical_csv)

# ----------------------------
# Sidebar menu
# ----------------------------
page = st.sidebar.radio("Navigate", ["📋 Session Builder", "👤 Player Variability", "📈 Weekly Progress"])

# ----------------------------
# Initialize session_state
# ----------------------------
if "selected_team" not in st.session_state:
    st.session_state.selected_team = "U23"
if "selected_date" not in st.session_state:
    st.session_state.selected_date = pd.to_datetime("2025-01-01")
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "session_total" not in st.session_state:
    st.session_state.session_total = None

# ----------------------------
# TAB 1: Session Builder
# ----------------------------
if page == "📋 Session Builder":
    st.title("⚽ Training Load Prediction App")
    st.write("Create a training session with multiple tasks and automatically predict physical demands.")

    st.subheader("📋 Session Info")
    teams = ["First Team", "B Team", "C Team", "U23", "U21", "U19"]

    col_team, col_date = st.columns(2)
    with col_team:
        st.session_state.selected_team = st.selectbox("Team", teams, index=teams.index(st.session_state.selected_team))
    with col_date:
        st.session_state.selected_date = pd.to_datetime(st.date_input("Date", st.session_state.selected_date))

    # Task input form
    st.subheader("➕ Add a New Task")
    col1, col2, col3 = st.columns(3)
    with col1:
        task_type = st.selectbox("Task Type",
                                 ['Possession', 'Conditioned Game', 'Transition', 'Passing Drill', 'Game / Warm-up'])
    with col2:
        sets = st.number_input("Number of Sets", 1, 20, 1)
    with col3:
        set_duration = st.number_input("Set Duration (min)", 1, 60, 5)

    col4, col5, col6 = st.columns(3)
    with col4:
        length = st.number_input("Field Length (m)", 10, 120, 30)
    with col5:
        width = st.number_input("Field Width (m)", 5, 80, 20)
    with col6:
        jokers = st.number_input("Jokers", 0, 5, 0)

    col7, col8, col9 = st.columns(3)
    with col7:
        players_team1 = st.number_input("Team 1 Players (incl. GK)", 1, 11, 6)
    with col8:
        players_team2 = st.number_input("Team 2 Players (incl. GK)", 1, 11, 6)
    with col9:
        goalkeepers = st.number_input("Goalkeepers", 0, 4, 2)

    total_players = players_team1 + players_team2 + jokers
    density = (length * width) / total_players if total_players > 0 else 0
    total_duration = sets * set_duration

    col10, col11 = st.columns(2)
    with col10:
        st.metric("Total Players", total_players)
    with col11:
        st.metric("Total Duration (min)", total_duration)

    if st.button("✅ Add Task"):
        st.session_state.tasks.append({
            "Team": st.session_state.selected_team,
            "Date": st.session_state.selected_date,
            "TaskType": task_type,
            "Length (m)": length,
            "Width (m)": width,
            "Players_Team1": players_team1,
            "Players_Team2": players_team2,
            "Jokers": jokers,
            "Goalkeepers": goalkeepers,
            "Total_Players": total_players,
            "Density (m2/player)": density,
            "Duration (min)": total_duration,
            "Sets": sets
        })
        st.success(f"Task '{task_type}' added for {st.session_state.selected_team}. Total tasks: {len(st.session_state.tasks)}")

    # Show added tasks
    if st.session_state.tasks:
        st.subheader("📋 Added Tasks")
        df_tasks = pd.DataFrame(st.session_state.tasks)
        df_tasks.index = df_tasks.index + 1
        for idx, row in df_tasks.iterrows():
            st.markdown(
                f"**{idx}. {row['TaskType']}** | "
                f"Team: {row['Team']} | "
                f"Date: {row['Date'].strftime('%Y-%m-%d')} | "
                f"Players: {row['Total_Players']} | "
                f"Duration: {row['Duration (min)']} min | "
                f"Field: {row['Length (m)']}x{row['Width (m)']} m"
            )
            delete_key = f"delete_{idx}"
            if st.button(f"❌ Delete Task {idx}", key=delete_key):
                st.session_state.tasks.pop(idx - 1)
                st.rerun()


    # Calculate session
    if st.session_state.tasks and st.button("🚀 Calculate Session"):
        all_preds = []
        for task in st.session_state.tasks:
            input_df = pd.DataFrame([task])
            input_prepared = preprocessor.transform(input_df)
            pred = rf.predict(input_prepared)
            pred_df = pd.DataFrame(pred, columns=output_columns)
            pred_df.insert(0, "Team", task["Team"])
            pred_df.insert(0, "TaskType", task["TaskType"])
            all_preds.append(pred_df)

        all_preds_df = pd.concat(all_preds, ignore_index=True)
        st.session_state.predictions = all_preds_df

        session_total = all_preds_df[output_columns].sum().to_frame(name="Session Total")

        team = st.session_state.selected_team
        session_date = st.session_state.selected_date
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])
        team_hist = df_hist[df_hist["Team"] == team]
        last_MD = team_hist[(team_hist["Date"] < session_date) & (team_hist["Duration (min)"] >= 60)]
        if not last_MD.empty:
            last_MD_mean = last_MD[output_columns].mean()
            percent_vs_last_MD = (session_total["Session Total"] / last_MD_mean) * 100
            session_total["% vs Last Match (>=60min)"] = percent_vs_last_MD
        else:
            session_total["% vs Last Match (>=60min)"] = np.nan

        st.session_state.session_total = session_total

        # Show results
        st.subheader("📊 Session Prediction (Per Task)")
        st.dataframe(all_preds_df)
        st.subheader("📈 Total Session Prediction")
        st.dataframe(session_total)

        # Plot % vs Last Match
        if not last_MD.empty:
            session_percentage_plot = percent_vs_last_MD.sort_values(ascending=False)
    
            # Asignar colores según % vs Last Match
            colors = []
            for x in session_percentage_plot:
                if x > 150 or x < 50:
                    colors.append("#B30808")  # rojo
                else:
                    colors.append("#084d9b")  # azul oscuro

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = session_percentage_plot.plot(kind='bar', ax=ax, color=colors)
            
            plt.ylabel("% vs Last Match")
            plt.xticks(rotation=45, ha="right")
            ax.set_xticklabels(session_percentage_plot.index)
            
            # Añadir leyenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#084d9b', label='50-150%'),
                Patch(facecolor='#B30808', label='<50% o >150%')
            ]
            ax.legend(handles=legend_elements, loc="upper right")
            
            st.pyplot(fig)

# ----------------------------
# TAB 2: Player Variability
# ----------------------------
elif page == "👤 Player Variability":
    st.subheader("👤 Player Variability (Last 10 Sessions)")

    if st.session_state.predictions is None:
        st.info("Please create and calculate a session in Session Builder first.")
    else:
        team = st.session_state.selected_team
        session_date = st.session_state.selected_date
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])
        team_hist = df_hist[df_hist["Team"] == team]
        past_sessions = team_hist[team_hist["Date"] < session_date]

        if past_sessions.empty:
            st.warning("Not enough historical data for last 10 sessions.")
        else:
            past_sessions["SessionKey"] = past_sessions["Team"] + "_" + past_sessions["Date"].dt.strftime('%Y-%m-%d')
            last10_sessions = past_sessions["SessionKey"].drop_duplicates().tail(10)
            last10 = past_sessions[past_sessions["SessionKey"].isin(last10_sessions)]
            session_means = last10.groupby(["SessionKey", "Date"])[output_columns].mean().reset_index()
            merged = last10.merge(session_means, on=["SessionKey", "Date"], suffixes=("", "_TeamMean"))

            for col in output_columns:
                merged[f"{col}_Perc"] = (merged[col] / merged[f"{col}_TeamMean"]) * 100

            player_variability = merged.groupby("Player")[[f"{c}_Perc" for c in output_columns]].mean()
            player_variability_diff = player_variability - 100
            player_variability_diff.columns = output_columns

            players = merged["Player"].unique()
            selected_player = st.selectbox("Select Player", players)

            st.write(f"📊 Average % deviation for {selected_player}")

            player_table = pd.DataFrame({
                "Difference from team's mean (%)": player_variability_diff.loc[selected_player].values
            }, index=output_columns)
            st.dataframe(player_table)

            # Plot
            plot_df = player_variability_diff.loc[selected_player].sort_values(ascending=False)
            colors = ["#B30808" if x < 0 else "#084d9b" for x in plot_df]  
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_df.plot(kind="bar", ax=ax, color=colors)
            plt.ylabel("Difference from team's mean (%)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            if st.checkbox("Show all players"):
                st.dataframe(player_variability_diff)

# ----------------------------
# TAB 3: Weekly Progress
# ----------------------------
elif page == "📈 Weekly Progress":
    st.subheader("📈 Weekly Progress")

    if st.session_state.predictions is None:
        st.info("Please create and calculate a session in Session Builder first.")
    else:
        team = st.session_state.selected_team
        session_date = st.session_state.selected_date
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])
        team_hist = df_hist[df_hist["Team"] == team]

        week_hist = team_hist[(team_hist["Date"] < session_date) & (team_hist["Date"] >= session_date - timedelta(days=6))]

        if week_hist.empty:
            st.warning("Not enough data in the 6 days before the session.")
        else:
            day_labels = week_hist[["Date","DayLabel"]].drop_duplicates()
            session_means = week_hist.groupby(["Team","Date"])[output_columns].mean().reset_index()
            session_means = session_means.merge(day_labels, on="Date", how="left")
            session_means = session_means.sort_values("Date")

            day_sequence = ["MD", "MD+1", "MD+2", "MD-4", "MD-3", "MD-2", "MD-1"]
            last_day_label = session_means.iloc[-1]["DayLabel"]
            last_idx = day_sequence.index(last_day_label)
            predicted_label = day_sequence[(last_idx + 1) % len(day_sequence)]

            pred_row = pd.DataFrame({
                "Team": [team],
                "Date": [session_date],
                **{col: [st.session_state.session_total.loc[col, "Session Total"]] for col in output_columns},
                "DayLabel": [predicted_label]
            })

            plot_df = pd.concat([session_means, pred_row], ignore_index=True)
            plot_df = plot_df.sort_values("Date")

            variable = st.selectbox("Select Variable", output_columns)

            colors = ["#AD9402" if x == predicted_label else "#084d9b" for x in plot_df["DayLabel"]]
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(plot_df["DayLabel"], plot_df[variable], color=colors)
            plt.ylabel(variable)
            plt.xlabel("Day")
            plt.xticks(rotation=45, ha="right")

            legend_elements = [Patch(facecolor='#084d9b', label='Team Avg'),
                               Patch(facecolor="#AD9402", label='Predicted Session')]
            ax.legend(handles=legend_elements, loc="upper left")

            st.pyplot(fig)


