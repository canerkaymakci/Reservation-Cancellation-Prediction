import streamlit as st
import pandas as pd
import datetime as dt
from datetime import datetime
import joblib

st.sidebar.write("Contact")
st.sidebar.write("Rabia Nur Gülmez")
st.sidebar.write("Caner Kaymakçı")

try:
    # TITLE
    st.markdown(
        """
        <style>
        .centered-title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
        }
        .centered-title-2 {
            text-align: center;
            font-size: 3em;
        }
        </style>
        <div class="centered-title">
            Miuuloria Resort
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="centered-title-2">
            Reservation
        </div>
        """,
        unsafe_allow_html=True
    )

    # DATE INPUT
    today = datetime.now()
    next_year = today.year + 1
    jan_1 = dt.date(today.year, 1, 1)
    dec_31 = dt.date(today.year, 12, 31)

    d = st.date_input(
        "Select Your Reservation",
        (datetime.now(), datetime.now() + dt.timedelta(days=7)),
        jan_1,
        dec_31,
        format="DD.MM.YYYY",
    )

    if len(d) == 2:
        start_date, end_date = d
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.min.time())

        day_difference = (start_date - datetime.now()).days
        reservation_length = (end_date - start_date).days
        arrival_day = start_date.day
        arrival_month = start_date.month

    elif len(d) == 1:
        st.write("Lütfen bitiş tarihini de seçin.")
    else:
        st.write("Lütfen başlangıç ve bitiş tarihlerini seçin.")

    col_left, col_right = st.columns(2)

    # NUMBER OF PEOPLE
    col_left.text_input("Number Of Adults", key="adult")
    col_left.text_input("Number Of Children", key="children")

    # ROOM TYPE
    room_type = ["Room Type 1", "Room Type 2", "Room Type 3", "Room Type 4", "Room Type 5", "Room Type 6",
                 "Room Type 7"]
    option = col_right.selectbox(
        'Choose a Room Type',
        ["Room Type 1", "Room Type 2", "Room Type 3", "Room Type 4", "Room Type 5", "Room Type 6", "Room Type 7"])
    room_type_index = room_type.index(option)

    # MEAL TYPE
    meal_plan = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
    option = col_right.selectbox(
        'Choose a Meal Plan',
        ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    meal_plan_index = meal_plan.index(option)

    # SPECIAL OPTIONS
    total_options = 0

    if col_left.checkbox('Fitness'):
        total_options += 1

    if col_left.checkbox('Spa'):
        total_options += 1

    if col_left.checkbox('Pool'):
        total_options += 1

    if col_right.checkbox('Room Service'):
        total_options += 1

    if col_right.checkbox('Social Activities'):
        total_options += 1

    # PARKING SPACE
    parking_space = 0
    if col_right.checkbox('Car Parking Space'):
        parking_space = 1
    else:
        parking_space = 0

    # ROOM PRICE
    st.text_input("Room Price", key="room_price")

    # MODEL + SCALER
    model = joblib.load("xgb.pkl")
    scalers = joblib.load("scalers.pkl")

    # INPUT DATA
    input_data = {
        "no_of_adults": [st.session_state.adult],
        "no_of_children": [st.session_state.children],
        "type_of_meal_plan": [meal_plan_index],
        "required_car_parking_space": [parking_space],
        "room_type_reserved": [room_type_index],
        "lead_time": [day_difference],
        "arrival_month": [arrival_month],
        "arrival_date": [arrival_day],
        "avg_price_per_room": [st.session_state.room_price],
        "no_of_special_requests": [total_options],
        "visit_duration": [reservation_length],
    }

    X = pd.DataFrame(input_data)

    # SCALING
    num_features = ["no_of_adults", "no_of_children", "required_car_parking_space", "lead_time", "arrival_month",
                    "arrival_date", "avg_price_per_room", "no_of_special_requests", "visit_duration"]

    for feature in num_features:
        X[feature] = scalers[feature].transform(X[[feature]].astype(float))

    if st.button("Predict", key="predict"):
        y_pred = model.predict(X)
        if y_pred[0] == 0:
            st.write("Customer Can Make Reservation")
            if st.button("Proceed"):
                st.session_state.page = "booking"
        else:
            st.write("Customer Can Not Make Reservation")
            if st.button("Return"):
                st.session_state.page = "main"

except Exception as e:
    st.write("")

