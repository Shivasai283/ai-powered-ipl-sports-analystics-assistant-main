import pandas as pd
import streamlit as st
from openai import OpenAI


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="IPL Sports Analytics Assistant",
    page_icon="🏏",
    layout="wide"
)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    df = pd.read_csv("ipl_matches.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean text columns
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load ipl_matches.csv: {e}")
    st.stop()


# ==================================================
# OPENAI CLIENT
# ==================================================

def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]

        if not api_key:
            return None

        return OpenAI(api_key=api_key)

    except Exception:
        return None


# ==================================================
# HEADER
# ==================================================

st.title("🏏 IPL Sports Analytics Assistant")

st.write(
    "Analyze IPL match data, check team performance, "
    "and ask questions about the available matches."
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Choose an option",
    [
        "Dashboard",
        "Team Analytics",
        "Match Data",
        "Ask Assistant"
    ]
)


# ==================================================
# BASIC INFORMATION
# ==================================================

total_matches = len(df)

if "team" in df.columns:
    total_teams = df["team"].nunique()
else:
    total_teams = 0

if "result" in df.columns:
    total_wins = (
        df["result"].str.lower() == "win"
    ).sum()
else:
    total_wins = 0


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("📊 IPL Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Matches",
            total_matches
        )

    with col2:
        st.metric(
            "Teams",
            total_teams
        )

    with col3:
        st.metric(
            "Recorded Wins",
            total_wins
        )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Rows:**", df.shape[0])

    with col2:
        st.write("**Columns:**", df.shape[1])

    st.write("### Available Columns")

    st.write(", ".join(df.columns))


# ==================================================
# TEAM ANALYTICS
# ==================================================

elif page == "Team Analytics":

    st.header("🏆 Team Analytics")

    if "team" not in df.columns:
        st.error("The CSV does not contain a 'team' column.")
        st.stop()

    teams = sorted(
        df["team"].dropna().unique()
    )

    selected_team = st.selectbox(
        "Select a team",
        teams
    )

    team_data = df[
        df["team"].str.lower() == selected_team.lower()
    ]

    st.subheader(
        f"Performance of {selected_team}"
    )

    total_team_matches = len(team_data)

    if "result" in df.columns:

        wins = (
            team_data["result"]
            .str.lower()
            .eq("win")
            .sum()
        )

        losses = (
            team_data["result"]
            .str.lower()
            .eq("lose")
            .sum()
        )

    else:
        wins = 0
        losses = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Matches",
            total_team_matches
        )

    with col2:
        st.metric(
            "Wins",
            wins
        )

    with col3:
        st.metric(
            "Losses",
            losses
        )

    st.divider()

    st.subheader("Opponent-wise Matches")

    if "opponent" in team_data.columns:

        opponent_count = (
            team_data["opponent"]
            .value_counts()
            .reset_index()
        )

        opponent_count.columns = [
            "Opponent",
            "Matches"
        ]

        st.dataframe(
            opponent_count,
            use_container_width=True
        )

    st.subheader("Team Matches")

    st.dataframe(
        team_data,
        use_container_width=True
    )


# ==================================================
# MATCH DATA
# ==================================================

elif page == "Match Data":

    st.header("📋 IPL Match Data")

    st.write(
        f"Total records available: **{len(df)}**"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    st.subheader("🔎 Filter Data")

    if "team" in df.columns:

        teams = sorted(
            df["team"].dropna().unique()
        )

        selected = st.selectbox(
            "Filter by Team",
            ["All"] + teams
        )

        if selected != "All":

            filtered_df = df[
                df["team"] == selected
            ]

            st.dataframe(
                filtered_df,
                use_container_width=True
            )


# ==================================================
# AI ASSISTANT
# ==================================================

elif page == "Ask Assistant":

    st.header("🤖 Ask the IPL Assistant")

    st.write(
        "Ask a question about the IPL data."
    )

    st.info(
        "The assistant answers using the data available "
        "in ipl_matches.csv."
    )

    question = st.text_input(
        "Enter your question",
        placeholder="Example: How many matches did KKR win?"
    )

    if st.button("Ask Assistant"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------
            # CHECK OPENAI KEY
            # ------------------------------------------

            client = get_openai_client()

            if client is None:

                st.error(
                    "OpenAI API key was not found. "
                    "Please check .streamlit/secrets.toml"
                )

                st.stop()


            # ------------------------------------------
            # PREPARE DATA FOR AI
            # ------------------------------------------

            data_context = df.to_string(
                index=False
            )


            # ------------------------------------------
            # AI PROMPT
            # ------------------------------------------

            prompt = f"""
You are an IPL Sports Analytics Assistant.

You must answer the user's question using ONLY
the IPL dataset provided below.

IMPORTANT RULES:

1. Do not invent statistics.
2. Do not invent scores.
3. Do not invent match details.
4. If the requested information is not present
   in the dataset, clearly say that the information
   is not available.
5. Give a short, clear answer.
6. You can calculate totals, wins, losses and
   other information that can be directly calculated
   from the dataset.
7. The dataset columns are:
   team, opponent, result

IPL DATASET:
----------------
{data_context}
----------------

USER QUESTION:
{question}

Answer the user based only on the dataset.
"""


            # ------------------------------------------
            # CALL OPENAI
            # ------------------------------------------

            try:

                with st.spinner(
                    "🤖 Analyzing the IPL data..."
                ):

                    response = client.responses.create(
                        model="gpt-5.6-luna",
                        input=prompt
                    )

                answer = response.output_text

                st.success(
                    "🤖 Assistant Response"
                )

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Unable to contact OpenAI: {e}"
                )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "IPL Sports Analytics Assistant | "
    "Python + Pandas + Streamlit + OpenAI"
)