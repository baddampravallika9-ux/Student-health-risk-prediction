import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Health Risk Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Users\PRAVALLIKA\Downloads\playground-series-s6e7"

DATA_PATH = os.path.join(
    BASE_PATH,
    "train.csv"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_PATH,
    "C__Users_PRAVALLIKA_Downloads_playground-series-s6e7_preprocessor1.pkl"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "C__Users_PRAVALLIKA_Downloads_playground-series-s6e7_Architecture1.keras"
)

IMAGE_PATH = os.path.join(
    BASE_PATH,
    "student.webp"
)

deployment = load_model(r"C:\Users\PRAVALLIKA\OneDrive\New folder\ANN1.keras")
@st.cache_resource
def load_model_objects():

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessing = pickle.load(f)

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    return preprocessing, model
# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


try:

    df = load_dataset()

except Exception as e:

    st.error("Could not load train.csv")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD PREPROCESSOR AND MODEL
# ============================================================

@st.cache_resource
def load_model_objects():

    with open(
        PREPROCESSOR_PATH,
        "rb"
    ) as f:

        preprocessing = pickle.load(f)

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    return preprocessing, model


try:

    preprocessing, model = load_model_objects()

except Exception as e:

    st.error(
        "Could not load model or preprocessor."
    )

    st.exception(e)
    st.stop()


# ============================================================
# HEADER
# ============================================================

if os.path.exists(IMAGE_PATH):

    col1, col2, col3 = st.columns(
        [1, 1, 1]
    )

    with col2:

        st.image(
            IMAGE_PATH,
            width=250
        )


st.markdown(
    """
    <h1 style="text-align:center;">
        🎓 Student Health Risk Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="text-align:center;">
        Classification + Regression
    </h3>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "at-risk",
    "fit",
    "unhealthy"
]


# ============================================================
# STUDENT INPUT
# ============================================================

st.subheader(
    "Enter Student Details"
)


col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    st.markdown(
        "### Physical Details"
    )

    sleep_duration = st.number_input(
        "Sleep Duration (hours)",
        min_value=1.0,
        max_value=12.0,
        value=7.0,
        step=0.1
    )

    heart_rate = st.number_input(
        "Heart Rate",
        min_value=40,
        max_value=150,
        value=70,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=23.0,
        step=0.1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    st.markdown(
        "### Activity Details"
    )

    calorie_expenditure = st.number_input(
        "Calorie Expenditure",
        min_value=0,
        max_value=5000,
        value=2000,
        step=1
    )

    step_count = st.number_input(
        "Step Count",
        min_value=0,
        max_value=30000,
        value=5000,
        step=1
    )

    exercise_duration = st.number_input(
        "Exercise Duration (minutes)",
        min_value=0,
        max_value=240,
        value=30,
        step=1
    )

    water_intake = st.number_input(
        "Water Intake (litres)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    )

    diet_type = st.selectbox(
        "Diet Type",
        ["veg", "balanced", "non-veg"]
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    st.markdown(
        "### Lifestyle Details"
    )

    stress_level = st.selectbox(
        "Stress Level",
        ["low", "medium", "high"]
    )

    sleep_quality = st.selectbox(
        "Sleep Quality",
        ["poor", "average", "good"]
    )

    physical_activity_level = st.selectbox(
        "Physical Activity Level",
        [
            "sedentary",
            "moderate",
            "active"
        ]
    )

    smoking_alcohol = st.selectbox(
        "Smoking / Alcohol",
        [
            "no",
            "occasional",
            "yes"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "male",
            "female",
            "other"
        ]
    )


# ============================================================
# ORIGINAL INPUT DATAFRAME
# ============================================================

input_df = pd.DataFrame({

    "sleep_duration": [
        sleep_duration
    ],

    "heart_rate": [
        heart_rate
    ],

    "bmi": [
        bmi
    ],

    "calorie_expenditure": [
        calorie_expenditure
    ],

    "step_count": [
        step_count
    ],

    "exercise_duration": [
        exercise_duration
    ],

    "water_intake": [
        water_intake
    ],

    "diet_type": [
        diet_type
    ],

    "stress_level": [
        stress_level
    ],

    "sleep_quality": [
        sleep_quality
    ],

    "physical_activity_level": [
        physical_activity_level
    ],

    "smoking_alcohol": [
        smoking_alcohol
    ],

    "gender": [
        gender
    ]

})


# ============================================================
# FEATURE ENGINEERING
# SAME LOGIC AS NOTEBOOK
# ============================================================

def create_risk_features(data):

    data = data.copy()


    # --------------------------------------------------------
    # BMI RISK
    # --------------------------------------------------------

    bmi_deviation = (

        np.maximum(
            0,
            18.5 - data["bmi"]
        )

        +

        np.maximum(
            0,
            data["bmi"] - 24.9
        )
    )

    data["bmi_risk"] = np.clip(
        bmi_deviation / 10.0,
        0,
        1
    )


    # --------------------------------------------------------
    # STRESS RISK
    # --------------------------------------------------------

    data["stress_risk"] = (
        data["stress_level"]
        .map({
            "low": 0.0,
            "medium": 0.5,
            "high": 1.0
        })
    )


    # --------------------------------------------------------
    # SLEEP DURATION RISK
    # --------------------------------------------------------

    sleep_gap = (

        np.maximum(
            0,
            7 - data["sleep_duration"]
        )

        +

        np.maximum(
            0,
            data["sleep_duration"] - 9
        )
    )

    data["sleep_duration_risk"] = np.clip(
        sleep_gap / 4.0,
        0,
        1
    )


    # --------------------------------------------------------
    # ACTIVITY RISK
    # --------------------------------------------------------

    step_risk = (
        1
        -
        np.clip(
            data["step_count"] / 10000.0,
            0,
            1
        )
    )

    exercise_risk = (
        1
        -
        np.clip(
            data["exercise_duration"] / 60.0,
            0,
            1
        )
    )

    calorie_risk = (
        1
        -
        np.clip(
            data["calorie_expenditure"] / 600.0,
            0,
            1
        )
    )

    data["activity_risk"] = (

        0.40 * step_risk

        +

        0.30 * exercise_risk

        +

        0.30 * calorie_risk
    )


    # --------------------------------------------------------
    # HEART RATE RISK
    # --------------------------------------------------------

    data["heart_rate_risk"] = np.clip(

        np.abs(
            data["heart_rate"] - 70
        ) / 20.0,

        0,
        1
    )


    # --------------------------------------------------------
    # WATER RISK
    # --------------------------------------------------------

    data["water_risk"] = (

        1
        -
        np.clip(
            data["water_intake"] / 3.0,
            0,
            1
        )
    )


    # --------------------------------------------------------
    # SLEEP QUALITY RISK
    # --------------------------------------------------------

    data["sleep_quality_risk"] = (
        data["sleep_quality"]
        .map({
            "poor": 1.0,
            "average": 0.5,
            "good": 0.0
        })
    )


    # --------------------------------------------------------
    # LIFESTYLE RISK
    # --------------------------------------------------------

    data["lifestyle_risk"] = (
        data["smoking_alcohol"]
        .map({
            "no": 0.0,
            "occasional": 0.5,
            "yes": 1.0
        })
    )


    return data


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 PREDICT HEALTH RISK",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # ADD ENGINEERED FEATURES
        # ----------------------------------------------------

        test_data = create_risk_features(
            input_df
        )


        # ----------------------------------------------------
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

        feature_columns = [

            "sleep_duration",
            "heart_rate",
            "bmi",
            "calorie_expenditure",
            "step_count",
            "exercise_duration",
            "water_intake",

            "diet_type",
            "stress_level",
            "sleep_quality",
            "physical_activity_level",
            "smoking_alcohol",
            "gender",

            "bmi_risk",
            "stress_risk",
            "sleep_duration_risk",
            "activity_risk",
            "heart_rate_risk",
            "water_risk",
            "sleep_quality_risk",
            "lifestyle_risk"

        ]


        test_data = test_data[
            feature_columns
        ]


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        test_processed = (
            preprocessing.transform(
                test_data
            )
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            test_processed,
            verbose=0
        )


        # ====================================================
        # IMPORTANT
        # ====================================================

        # The notebook model MUST return two outputs:
        #
        # prediction[0] = classification
        # prediction[1] = regression


        if not isinstance(
            prediction,
            (list, tuple)
        ):

            st.error(
                "Your loaded .keras model has only ONE output."
            )

            st.write(
                "Prediction shape:",
                np.asarray(
                    prediction
                ).shape
            )

            st.write(
                "Model output:",
                model.output
            )

            st.warning(
                """
                The model saved in Architecture1.keras is
                not the same two-output model used in your
                notebook.

                Save the final two-output model again from
                your notebook and replace Architecture1.keras.
                """
            )

            st.stop()


        if len(prediction) != 2:

            st.error(
                "Model does not contain two outputs."
            )

            st.write(
                "Number of outputs:",
                len(prediction)
            )

            st.stop()


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        class_probability = np.asarray(
            prediction[0]
        )

        class_index = np.argmax(
            class_probability,
            axis=1
        )

        predicted_class = class_names[
            class_index[0]
        ]

        confidence = np.max(
            class_probability,
            axis=1
        )[0]


        # ====================================================
        # REGRESSION
        # ====================================================

        regression_prediction = np.asarray(
            prediction[1]
        )

        health_risk_score = float(
            regression_prediction
            .flatten()[0]
        )


        # The notebook's regression target is 0-100.
        # Therefore DO NOT multiply it by 100.

        health_risk_score = np.clip(
            health_risk_score,
            0,
            100
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


        # ----------------------------------------------------
        # CLASSIFICATION RESULT
        # ----------------------------------------------------

        with result_col1:

            st.markdown(
                "### Health Condition"
            )

            if predicted_class == "fit":

                st.success(
                    f"🟢 {predicted_class.upper()}"
                )

            elif predicted_class == "at-risk":

                st.warning(
                    f"🟡 {predicted_class.upper()}"
                )

            else:

                st.error(
                    f"🔴 {predicted_class.upper()}"
                )


            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ----------------------------------------------------
        # REGRESSION RESULT
        # ----------------------------------------------------

        with result_col2:

            st.markdown(
                "### Health Risk Score"
            )

            st.metric(
                "Risk Score",
                f"{health_risk_score:.2f} / 100"
            )


            if health_risk_score < 30:

                st.success(
                    "🟢 Low Health Risk"
                )

            elif health_risk_score < 60:

                st.warning(
                    "🟡 Moderate Health Risk"
                )

            else:

                st.error(
                    "🔴 High Health Risk"
                )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )


        probability_df = pd.DataFrame({

            "Health Condition": class_names,

            "Probability": [

                f"{prob * 100:.2f}%"

                for prob in class_probability[0]

            ]

        })


        st.table(
            probability_df
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)