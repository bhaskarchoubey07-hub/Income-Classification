from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from model import (
    DEFAULT_PREDICTION_FIELDS,
    REQUIRED_COLUMNS,
    build_prediction_row,
    get_data_profile,
    get_feature_importance,
    run_training_pipeline,
    validate_dataset,
)


st.set_page_config(
    page_title="Income Classification Dashboard",
    page_icon="IC",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path("data") / "income_evaluation.csv"
TARGET_COLUMN = "income"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(31, 119, 180, 0.16), transparent 35%),
                    radial-gradient(circle at top right, rgba(46, 204, 113, 0.16), transparent 32%),
                    linear-gradient(180deg, #f5f7fb 0%, #eef3f8 100%);
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }
            .hero-card {
                padding: 1.4rem 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(12, 18, 34, 0.92), rgba(27, 44, 75, 0.9));
                color: white;
                box-shadow: 0 20px 45px rgba(14, 25, 40, 0.18);
                margin-bottom: 1rem;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 0.9rem;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_default_dataset()


def build_overview_tab(df: pd.DataFrame, profile: dict) -> None:
    st.subheader("Data Overview")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", profile["shape"][0])
    metric_columns[1].metric("Columns", profile["shape"][1])
    metric_columns[2].metric("Missing Values", int(profile["missing_summary"]["missing_count"].sum()))
    metric_columns[3].metric("Numeric Features", len(profile["numeric_columns"]))

    left, right = st.columns((1.35, 1))

    with left:
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with right:
        with st.expander("Dataset Shape", expanded=True):
            st.write(f"Rows: `{profile['shape'][0]}`")
            st.write(f"Columns: `{profile['shape'][1]}`")

        with st.expander("Missing Value Summary", expanded=True):
            st.dataframe(profile["missing_summary"], use_container_width=True)

        with st.expander("Column Types", expanded=True):
            st.dataframe(profile["column_types"], use_container_width=True)


def build_visualization_tab(df: pd.DataFrame) -> None:
    st.subheader("Interactive Visualizations")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        age_fig = px.histogram(
            df,
            x="age",
            nbins=20,
            color_discrete_sequence=["#1f77b4"],
            title="Age Distribution",
        )
        age_fig.update_layout(template="plotly_white")
        st.plotly_chart(age_fig, use_container_width=True)

        income_fig = px.histogram(
            df,
            x="income",
            color="income",
            title="Income Distribution",
            color_discrete_sequence=["#4c78a8", "#f58518"],
        )
        income_fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(income_fig, use_container_width=True)

        marital_fig = px.histogram(
            df,
            x="marital-status",
            color="income",
            barmode="group",
            title="Marital Status vs Income",
            color_discrete_sequence=["#54a24b", "#e45756"],
        )
        marital_fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
        st.plotly_chart(marital_fig, use_container_width=True)

    with chart_right:
        education_fig = px.histogram(
            df,
            y="education",
            color="income",
            title="Education Distribution",
            color_discrete_sequence=["#72b7b2", "#ff9da6"],
        )
        education_fig.update_layout(template="plotly_white")
        st.plotly_chart(education_fig, use_container_width=True)

        hours_fig = px.histogram(
            df,
            x="hours-per-week",
            nbins=20,
            color_discrete_sequence=["#2ca02c"],
            title="Hours per Week Histogram",
        )
        hours_fig.update_layout(template="plotly_white")
        st.plotly_chart(hours_fig, use_container_width=True)

        corr_df = df.select_dtypes(include="number")
        corr_fig = px.imshow(
            corr_df.corr(numeric_only=True),
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Correlation Heatmap",
        )
        corr_fig.update_layout(template="plotly_white")
        st.plotly_chart(corr_fig, use_container_width=True)


def build_model_tab(training_artifacts: dict) -> None:
    st.subheader("Random Forest Model Performance")

    metrics_left, metrics_right, metrics_third = st.columns(3)
    metrics_left.metric("Accuracy", f"{training_artifacts['accuracy']:.2%}")
    metrics_right.metric("Train Size", training_artifacts["train_shape"][0])
    metrics_third.metric("Test Size", training_artifacts["test_shape"][0])

    result_left, result_right = st.columns((0.95, 1.05))

    with result_left:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=training_artifacts["confusion_matrix"],
            display_labels=training_artifacts["class_labels"],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Prediction vs Actual")
        st.pyplot(fig)
        plt.close(fig)

    with result_right:
        st.markdown("#### Classification Report")
        st.dataframe(training_artifacts["classification_report"], use_container_width=True)


def build_feature_importance_tab(feature_importance: pd.DataFrame) -> None:
    st.subheader("Feature Importance")
    importance_fig = px.bar(
        feature_importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="importance",
        color_continuous_scale="Teal",
    )
    importance_fig.update_layout(template="plotly_white", yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(importance_fig, use_container_width=True)


def build_prediction_tab(df: pd.DataFrame, training_artifacts: dict) -> None:
    st.subheader("Prediction Panel")
    st.caption("Categorical fields are auto-filled using the most common values from the training data.")

    defaults = training_artifacts["numeric_defaults"]
    input_columns = st.columns(len(DEFAULT_PREDICTION_FIELDS))
    user_inputs = {}

    for index, field in enumerate(DEFAULT_PREDICTION_FIELDS):
        user_inputs[field] = input_columns[index].number_input(
            field.replace("-", " ").title(),
            min_value=0.0,
            value=float(defaults.get(field, 0)),
            step=1.0,
        )

    if st.button("Predict Income Category", type="primary", use_container_width=False):
        input_frame = build_prediction_row(df, training_artifacts["feature_columns"], user_inputs)
        prediction = training_artifacts["model"].predict(input_frame)[0]
        probabilities = training_artifacts["model"].predict_proba(input_frame)[0]
        probability_map = dict(zip(training_artifacts["class_labels"], probabilities))

        result_left, result_right = st.columns((0.8, 1.2))
        with result_left:
            st.success(f"Predicted income: `{prediction}`")
        with result_right:
            prob_fig = px.bar(
                x=list(probability_map.keys()),
                y=list(probability_map.values()),
                labels={"x": "Income Class", "y": "Probability"},
                title="Prediction Confidence",
                color=list(probability_map.values()),
                color_continuous_scale="Viridis",
            )
            prob_fig.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(prob_fig, use_container_width=True)


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">Income Classification Dashboard</h1>
            <p style="margin:0;font-size:1.02rem;">
                Explore demographic patterns, train a Random Forest classifier, and predict whether
                annual income falls above or below the 50K threshold.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Dashboard Controls")
    uploaded_file = st.sidebar.file_uploader("Upload income dataset", type=["csv"])

    if not DATA_PATH.exists() and uploaded_file is None:
        st.error(
            "Default dataset not found at `data/income_evaluation.csv`. Upload a CSV file to continue."
        )
        st.stop()

    df = load_dataset(uploaded_file)

    try:
        validate_dataset(df, REQUIRED_COLUMNS)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    profile = get_data_profile(df)
    st.sidebar.markdown("### Dataset Information")
    st.sidebar.write(f"Rows: `{profile['shape'][0]}`")
    st.sidebar.write(f"Columns: `{profile['shape'][1]}`")
    st.sidebar.write(f"Missing values: `{int(profile['missing_summary']['missing_count'].sum())}`")

    available_features = [column for column in df.columns if column != TARGET_COLUMN]
    selected_features = st.sidebar.multiselect(
        "Feature Selection",
        options=available_features,
        default=available_features,
    )

    if not selected_features:
        st.warning("Select at least one feature to train the model.")
        st.stop()

    model_df = df[selected_features + [TARGET_COLUMN]].copy()
    training_artifacts = run_training_pipeline(model_df, TARGET_COLUMN)
    feature_importance = get_feature_importance(training_artifacts["model"], training_artifacts["feature_names"])

    overview_tab, viz_tab, model_tab, importance_tab, prediction_tab = st.tabs(
        [
            "Data Overview",
            "Visualizations",
            "Model Results",
            "Feature Importance",
            "Prediction Panel",
        ]
    )

    with overview_tab:
        build_overview_tab(df, profile)

    with viz_tab:
        build_visualization_tab(df)

    with model_tab:
        build_model_tab(training_artifacts)

    with importance_tab:
        build_feature_importance_tab(feature_importance)

    with prediction_tab:
        build_prediction_tab(df, training_artifacts)

    st.caption("Run locally with `streamlit run app.py`.")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
