import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import streamlit as st

# Title
st.title("Vehicle Routing Problem (VRP) - Dynamic Input")
st.write("Input depot and customer coordinates, then visualize the solution.")

# Define input fields for depot coordinates
st.header("Depot Coordinates")
x_depot = st.number_input("Depot X Coordinate:", min_value=0, max_value=100, step=1, value=0, key="depot_x")
y_depot = st.number_input("Depot Y Coordinate:", min_value=0, max_value=100, step=1, value=0, key="depot_y")

depot_coords = (x_depot, y_depot)

# Define input fields for customer coordinates
st.header("Customer Coordinates")
num_customers = st.number_input("Number of Customers:", min_value=1, max_value=20, step=1, value=10, key="num_customers")

customer_coords = []
for i in range(1, num_customers + 1):
    x_customer = st.number_input(f"Customer {i} X Coordinate:", min_value=0, max_value=100, step=1, key=f"customer_{i}_x")
    y_customer = st.number_input(f"Customer {i} Y Coordinate:", min_value=0, max_value=100, step=1, key=f"customer_{i}_y")
    customer_coords.append((x_customer, y_customer))

# Number of vehicles
num_vehicles = st.number_input("Number of Vehicles:", min_value=1, max_value=10, step=1, value=3, key="num_vehicles")

# Define color palette for vehicles
colors = sns.color_palette("tab10", num_vehicles)

# Button to compute and display solution
if st.button("Generate VRP Solution"):
    # Randomly assign customers to vehicles for demonstration purposes
    vehicle_assignments = [[] for _ in range(num_vehicles)]
    for customer in customer_coords:
        vehicle_index = random.randint(0, num_vehicles - 1)
        vehicle_assignments[vehicle_index].append(customer)

    # Plot the solution
    fig, ax = plt.subplots()

    # Plot depot
    ax.scatter(*depot_coords, c="red", marker="X", s=200, label="Depot", zorder=3)

    # Plot customers and vehicle routes
    for vehicle_idx, assigned_customers in enumerate(vehicle_assignments):
        if assigned_customers:
            color = colors[vehicle_idx]
            x_route = [depot_coords[0]] + [c[0] for c in assigned_customers] + [depot_coords[0]]
            y_route = [depot_coords[1]] + [c[1] for c in assigned_customers] + [depot_coords[1]]

            ax.plot(x_route, y_route, marker="o", label=f"Vehicle {vehicle_idx + 1}", color=color)
            ax.scatter(
                [c[0] for c in assigned_customers],
                [c[1] for c in assigned_customers],
                c=[color],
                s=100,
                zorder=3,
            )

    ax.set_title("Vehicle Routing Problem - Solution Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
