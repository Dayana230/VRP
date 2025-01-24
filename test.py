import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Title
st.title("Vehicle Routing Problem: Customer Coordinates")
st.write("Enter or modify the coordinates (x, y) and demands for up to 20 customers. Default depot is set at (0, 0).")

# Default data
default_data = {
    "Customer": [f"C{i}" for i in range(1, 21)],
    "X_Coordinate": [5, 10, 15, 7, 18, 22, 25, 30, 35, 40, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68],
    "Y_Coordinate": [10, 15, 5, 20, 25, 8, 12, 18, 30, 5, 10, 15, 25, 35, 40, 45, 12, 8, 5, 20],
    "Demand": [2, 4, 6, 8, 3, 5, 7, 9, 2, 3, 5, 6, 4, 7, 2, 3, 8, 5, 6, 7],
}

# Convert to DataFrame
df = pd.DataFrame(default_data)

# Add a depot at (0, 0)
depot_x, depot_y = 0, 0

# Editable table for user input
st.write("Modify the customer data below:")
edited_df = st.data_editor(df, num_rows="dynamic", key="customer_table")

# Extract data from edited DataFrame
customers = edited_df["Customer"].tolist()
x_coords = edited_df["X_Coordinate"].tolist()
y_coords = edited_df["Y_Coordinate"].tolist()
demands = edited_df["Demand"].tolist()

# Display coordinates on a plot
if st.button("Plot Customer and Depot Locations"):
    # Define color palette
    colors = sns.color_palette("pastel", len(customers))
    
    # Plot the depot
    fig, ax = plt.subplots()
    ax.scatter(depot_x, depot_y, c="red", s=150, label="Depot", zorder=3)
    ax.annotate("Depot", (depot_x, depot_y), fontsize=12, ha='center', va='bottom')

    # Plot each customer
    for i, (customer, x, y) in enumerate(zip(customers, x_coords, y_coords)):
        ax.scatter(x, y, c=[colors[i]], s=120, label=f"{customer} (Demand: {demands[i]})", zorder=2)
        ax.annotate(customer, (x, y), fontsize=10, ha='center', va='bottom')
    
    # Customize plot
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Customer and Depot Locations")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(12, 8)
    st.pyplot(fig)
