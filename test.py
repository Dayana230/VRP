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
st.write("### Input Customer Details")
edited_data = []
columns = ["Customer_ID", "X_Coordinate", "Y_Coordinate", "Demand"]
for idx, row in st.session_state.customer_data.iterrows():
    cols = st.columns([1, 1, 1, 1])
    customer_id = cols[0].text_input(
        f"Customer {idx + 1}", value=int(row["Customer_ID"]), key=f"customer_id_{idx}", disabled=True
    )
    x_coord = cols[1].number_input(
        f"X (Customer {idx + 1})", value=float(row["X_Coordinate"]), step=1.0, key=f"x_coord_{idx}"
    )
    y_coord = cols[2].number_input(
        f"Y (Customer {idx + 1})", value=float(row["Y_Coordinate"]), step=1.0, key=f"y_coord_{idx}"
    )
    demand = cols[3].number_input(
        f"Demand (Customer {idx + 1})", value=int(row["Demand"]), step=1, key=f"demand_{idx}"
    )
    edited_data.append({"Customer_ID": int(customer_id), "X_Coordinate": x_coord, "Y_Coordinate": y_coord, "Demand": demand})

# Update session state with edited data
st.session_state.customer_data = pd.DataFrame(edited_data)

# Button to add a new customer
if st.button("Add Customer"):
    new_customer = {
        "Customer_ID": len(st.session_state.customer_data) + 1,
        "X_Coordinate": 0,
        "Y_Coordinate": 0,
        "Demand": 5
    }
    st.session_state.customer_data = pd.concat(
        [st.session_state.customer_data, pd.DataFrame([new_customer])], ignore_index=True
    )

# Display customer data table
st.write("### Collected Customer Data")
st.dataframe(st.session_state.customer_data)

# Convert customer data into a dictionary
customers = st.session_state.customer_data.set_index('Customer_ID').T.to_dict()

depot = (0, 0)

# Helper functions (same as in the original code)
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

def initialize_population(customers, num_vehicles):
    population = []
    customer_ids = list(customers.keys())
    for _ in range(population_size):
        random.shuffle(customer_ids)
        solution = []
        current_route = []
        current_load = 0
        for customer_id in customer_ids:
            demand = customers[customer_id]['Demand']
            if current_load + demand <= vehicle_capacity:
                current_route.append(customer_id)
                current_load += demand
            else:
                solution.append(current_route)
                current_route = [customer_id]
                current_load = demand
        if current_route:
            solution.append(current_route)
        population.append(solution)
    return population

def select_parents(population, fitness_values):
    parents = []
    for _ in range(2):
        competitors = random.sample(range(len(population)), 5)
        best = min(competitors, key=lambda idx: fitness_values[idx])
        parents.append(population[best])
    return parents

def crossover(parent1, parent2):
    child = []
    for route1, route2 in zip(parent1, parent2):
        midpoint = len(route1) // 2
        child_route = route1[:midpoint] + [c for c in route2 if c not in route1[:midpoint]]
        child.append(child_route)
    return child

def mutate(solution):
    for route in solution:
        if random.random() < mutation_rate and len(route) > 1:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
    return solution

def genetic_algorithm(customers, num_vehicles):
    population = initialize_population(customers, num_vehicles)

    best_solution = None
    best_distance = float('inf')
    fitness_history = []

    for generation in range(num_generations):
        fitness_values = [calculate_total_distance(sol, customers) for sol in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_values)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        population = new_population

        best_solution = min(population, key=lambda sol: calculate_total_distance(sol, customers))
        best_distance = calculate_total_distance(best_solution, customers)
        fitness_history.append(best_distance)

        st.write(f"Generation {generation + 1}: Best Distance = {best_distance}")

        if best_distance <= target_fitness:
            st.success(f"Target fitness achieved in generation {generation + 1}!")
            return best_solution, best_distance, fitness_history

    st.warning("Target fitness not achieved within the generations.")
    return best_solution, best_distance, fitness_history

if st.button("Run Genetic Algorithm"):
    best_solution, best_distance, fitness_history = genetic_algorithm(customers, num_vehicles)
    st.write("Best Solution:", best_solution)
    st.write("Best Distance:", best_distance)

    def plot_solution(solution, customers, depot):
        plt.figure(figsize=(10, 8))
        plt.scatter(depot[0], depot[1], color='red', s=100, marker='X', label="Depot")

        vehicle_colors = ['green', 'purple', 'orange', 'brown', 'cyan', 'blue', 'pink', 'yellow']
        legend_handles = [mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label="Depot")]

        for idx, route in enumerate(solution):
            color = vehicle_colors[idx % len(vehicle_colors)]
            current_location = depot
            for customer_id in route:
                customer = customers[customer_id]
                plt.scatter(customer['X_Coordinate'], customer['Y_Coordinate'], color=color, s=50)
                plt.text(customer['X_Coordinate'], customer['Y_Coordinate'], f'C{customer_id}', fontsize=9)
                plt.plot([current_location[0], customer['X_Coordinate']],
                         [current_location[1], customer['Y_Coordinate']], color=color, linestyle='-', marker='o')
                current_location = (customer['X_Coordinate'], customer['Y_Coordinate'])
            plt.plot([current_location[0], depot[0]], [current_location[1], depot[1]], color=color, linestyle='-', marker='X')
            legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='-', markersize=6, label=f"Vehicle {idx + 1}"))

        plt.title("Vehicle Routing Problem - Solution Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(handles=legend_handles, loc="best")
        plt.grid(True)
        st.pyplot(plt)

    plot_solution(best_solution, customers, depot)
