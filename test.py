import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random

# Streamlit app
st.title("Vehicle Routing Problem - Genetic Algorithm")

# Sidebar for parameters
st.sidebar.header("Genetic Algorithm Parameters")
vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", value=15, step=1)
num_generations = st.sidebar.number_input("Number of Generations", value=100, step=1)
population_size = st.sidebar.number_input("Population Size", value=50, step=1)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
target_fitness = st.sidebar.number_input("Target Fitness", value=950, step=1)
num_vehicles = st.sidebar.number_input("Number of Vehicles", value=5, step=1)

# Default customer data
default_data = [
    {"Customer_ID": 1, "X_Coordinate": 5, "Y_Coordinate": 10, "Demand": 2},
    {"Customer_ID": 2, "X_Coordinate": 10, "Y_Coordinate": 15, "Demand": 4},
    {"Customer_ID": 3, "X_Coordinate": 15, "Y_Coordinate": 5, "Demand": 6},
    {"Customer_ID": 4, "X_Coordinate": 7, "Y_Coordinate": 20, "Demand": 8},
    {"Customer_ID": 5, "X_Coordinate": 18, "Y_Coordinate": 25, "Demand": 3},
]

# Initialize customer data
if "customer_data" not in st.session_state:
    st.session_state.customer_data = pd.DataFrame(default_data)

# Layout for customer data table inputs
st.write("### Customer Details")
edited_data = []
columns = ["Customer_ID", "X_Coordinate", "Y_Coordinate", "Demand"]
for idx, row in st.session_state.customer_data.iterrows():
    cols = st.columns([1, 1, 1, 1])
    cols[0].markdown(f"**Customer {int(row['Customer_ID'])}**")  # Display Customer ID as a title, not editable
    x_coord = cols[1].number_input(
        "Coordinate X", value=float(row["X_Coordinate"]), step=1.0, key=f"x_coord_{idx}"
    )
    y_coord = cols[2].number_input(
        "Coordinate Y", value=float(row["Y_Coordinate"]), step=1.0, key=f"y_coord_{idx}"
    )
    demand = cols[3].number_input(
        "Demand", value=int(row["Demand"]), step=1, key=f"demand_{idx}"
    )
    edited_data.append({"Customer_ID": int(row["Customer_ID"]), "X_Coordinate": x_coord, "Y_Coordinate": y_coord, "Demand": demand})

# Update session state with edited data
st.session_state.customer_data = pd.DataFrame(edited_data)

# Add a new customer functionality
if st.button("Add Customer"):
    new_customer_id = len(st.session_state.customer_data) + 1
    new_customer = {
        "Customer_ID": new_customer_id,
        "X_Coordinate": 0,
        "Y_Coordinate": 0,
        "Demand": 5,
    }
    st.session_state.customer_data = pd.concat(
        [st.session_state.customer_data, pd.DataFrame([new_customer])], ignore_index=True
    )

# Helper functions remain unchanged
depot = (0, 0)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_total_distance(solution, customers):
    total_distance = 0
    for route in solution:
        current_location = depot
        for customer_id in route:
            customer = customers[customer_id]
            total_distance += calculate_distance(current_location, (customer['X_Coordinate'], customer['Y_Coordinate']))
            current_location = (customer['X_Coordinate'], customer['Y_Coordinate'])
        total_distance += calculate_distance(current_location, depot)
    return total_distance

# Other unchanged functions (initialize_population, selection, crossover, mutation, genetic_algorithm, plot_solution)
# If you need me to modify anything else, let me know!

# Display collected data
st.write("### Collected Customer Data")
st.dataframe(st.session_state.customer_data)
