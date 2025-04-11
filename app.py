import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("IPL Team Score Predictor (CSV-Based)")

st.write("""
Upload a CSV file with IPL ball-by-ball data. The app will train a simple model 
on-the-fly to predict the **final innings score** based on team and innings info.
""")

uploaded_file = st.file_uploader("Upload IPL CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Basic cleaning
    df["extra_type"].fillna("None", inplace=True)
    df["player_out"].fillna("None", inplace=True)
    df["kind"].fillna("None", inplace=True)
    df["fielders_involved"].fillna("None", inplace=True)

    # Add innings total per match/team
    df["ball_id"] = df.groupby(["ID", "innings"]).cumcount() + 1
    innings_summary = df.groupby(["ID", "BattingTeam", "innings"]).agg({
        "total_run": "sum",
        "ball_id": "max",
        "isWicketDelivery": "sum"
    }).reset_index()

    innings_summary.rename(columns={
        "total_run": "final_score",
        "ball_id": "total_balls",
        "isWicketDelivery": "wickets_lost"
    }, inplace=True)

    # Feature encoding
    team_names = innings_summary["BattingTeam"].unique()
    team = st.selectbox("Select Batting Team", team_names)
    inning = st.selectbox("Select Innings", [1, 2])

    # Model training
    X = innings_summary[["total_balls", "wickets_lost", "innings"]]
    y = innings_summary["final_score"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict
    if st.button("Predict Score"):
        # Estimate a sample input (example: team in 2nd innings with average stats)
        team_data = innings_summary[
            (innings_summary["BattingTeam"] == team) &
            (innings_summary["innings"] == inning)
        ]

        if not team_data.empty:
            avg_balls = int(team_data["total_balls"].mean())
            avg_wickets = int(team_data["wickets_lost"].mean())

            input_df = pd.DataFrame([[avg_balls, avg_wickets, inning]],
                                    columns=["total_balls", "wickets_lost", "innings"])
            prediction = int(model.predict(input_df)[0])

            st.subheader("Prediction Result")
            st.write(f"**Team:** {team}")
            st.write(f"**Innings:** {inning}")
            st.write(f"**Predicted Final Score:** {prediction} runs")
        else:
            st.warning("Not enough data for selected team and innings.")

else:
    st.info("Please upload a CSV file to get started.")
