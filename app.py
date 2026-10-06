
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Monthly Store Demand Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load best_model.pkl")
    st.info(
        "Make sure best_model.pkl is in the same folder as app.py."
    )
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Monthly Store Demand Prediction")

st.write(
    "Predict the expected monthly sales demand for a store "
    "using a machine learning regression model."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter Store Information")


col1, col2 = st.columns(2)


with col1:

    store = st.number_input(
        "Store ID",
        min_value=1,
        max_value=1115,
        value=1,
        step=1
    )

    year = st.number_input(
        "Year",
        min_value=2013,
        max_value=2030,
        value=2015,
        step=1
    )

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: pd.Timestamp(
            year=2000,
            month=x,
            day=1
        ).strftime("%B")
    )

    average_customers = st.number_input(
        "Average Customers",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    total_promo_days = st.number_input(
        "Total Promo Days",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


with col2:

    competition_distance = st.number_input(
        "Average Competition Distance",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    store_type = st.selectbox(
        "Store Type",
        ["a", "b", "c", "d"]
    )

    assortment = st.selectbox(
        "Assortment",
        ["a", "b", "c"]
    )

    state_holiday = st.selectbox(
        "State Holiday",
        ["0", "a", "b", "c"]
    )

    school_holiday = st.number_input(
        "School Holiday Days",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    open_days = st.number_input(
        "Open Days",
        min_value=0.0,
        max_value=31.0,
        value=26.0,
        step=1.0
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "Store": [store],
    "Year": [year],
    "Month": [month],
    "Average_Customers": [average_customers],
    "Total_Promo_Days": [total_promo_days],
    "Average_CompetitionDistance": [competition_distance],
    "StoreType": [store_type],
    "Assortment": [assortment],
    "StateHoliday": [state_holiday],
    "SchoolHoliday": [school_holiday],
    "Open_Days": [open_days]
})


# ============================================================
# SHOW INPUT DATA
# ============================================================

with st.expander("🔍 View Input Data"):
    st.dataframe(input_data)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Monthly Demand",
    use_container_width=True
):

    try:

        prediction = model.predict(input_data)

        predicted_demand = prediction[0]

        # Demand cannot be negative
        predicted_demand = max(0, predicted_demand)

        st.success("Prediction generated successfully!")

        st.metric(
            label="Predicted Monthly Demand",
            value=f"{predicted_demand:,.0f} units"
        )

        st.write(
            f"### 📦 Expected Demand: "
            f"**{predicted_demand:,.0f} units**"
        )

        st.caption(
            "This prediction is generated using the best regression "
            "model selected during training."
        )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.write("Error details:")
        st.code(str(e))


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About the Model"):

    st.write(
        """
        This application uses a machine learning regression model
        trained on the Rossmann Store Sales dataset.

        The Colab notebook evaluates:

        • Multiple Linear Regression
        • Ridge Regression
        • Lasso Regression
        • Decision Tree Regression
        • Random Forest Regression

        The model with the lowest RMSE is selected as the final model.

        The saved best_model.pkl contains both the preprocessing
        pipeline and the trained machine learning model.
        """
    )

st.caption(
    "Monthly Store Demand Prediction | Machine Learning Project"
)
