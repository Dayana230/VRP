import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Streamlit app title
st.title("Vehicle Routing Problem - Genetic Algorithm")

# Sidebar for parameters
st.sidebar.header("Genetic Algorithm Parameters")
vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", value=15, step=1)
num_generations = st.sidebar.number_input("Number of Generations", value=100, step=1)
population_size = st.sidebar.number_input("Population Size", value=50, step=1)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
target_fitness = st.sidebar.number_input("Target Fitness", value=950, step=1)
num_vehicles = st.sidebar.number_input("Number of Vehicles", value=5, step=1)

# Initialize session state for customers data if not already initialized
if "customers" not in st.session_state:
    st.session_state.customers = []

# Function to add a new customer to the list
def add_customer():
    # Assign customer ID automatically
    customer_id = f"Customer {len(st.session_state.customers) + 1}"

    # Input fields for customer data in a row
    cols = st.columns([2, 1, 1, 1])  # 4 columns for Customer ID, X, Y, and Demand
    with cols[0]:
        st.write(customer_id)
    with cols[1]:
        x_coord = st.number_input(f"x-coordinate ({customer_id})", min_value=-100, max_value=100, step=1, key=f"x_coord_{customer_id}")
    with cols[2]:
        y_coord = st.number_input(f"y-coordinate ({customer_id})", min_value=-100, max_value=100, step=1, key=f"y_coord_{customer_id}")
    with cols[3]:
        demand = st.number_input(f"Demand ({customer_id})", min_value=1, step=1, key=f"demand_{customer_id}")
    
    # Store customer data if all fields are filled
    if x_coord and y_coord and demand:
        st.session_state.customers.append({
            "Customer_ID": customer_id,
            "X_Coordinate": x_coord,
            "Y_Coordinate": y_coord,
            "Demand": demand
        })

# Initially add the first 5 customers
if len(st.session_state.customers) < 5:
    add_customer()

# Button to add more customers dynamically
if st.button("Add Customer"):
    add_customer()

# Create a layout with a table header and rows for each customer
if len(st.session_state.customers) > 0:
    st.write("Customer Data:")
    cols = st.columns([2, 1, 1, 1])  # 4 columns for Customer ID, X, Y, and Demand
    
    # Header row for the table
    with cols[0]:
        st.write("Customer ID")
    with cols[1]:
        st.write("X Coordinate")
    with cols[2]:
        st.write("Y Coordinate")
    with cols[3]:
        st.write("Demand")
    
    # Rows for each customer
    for customer in st.session_state.customers:
        with cols[0]:
            st.write(customer['Customer_ID'])
        with cols[1]:
            st.write(customer['X_Coordinate'])
        with cols[2]:
            st.write(customer['Y_Coordinate'])
        with cols[3]:
            st.write(customer['Demand'])

# Option to clear all customer data
if st.button("Clear All Customers"):
    st.session_state.customers = []
    st.success("All customer data cleared.")

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Genetic Algorithm functions
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

def calculate_total_distance(solution, customers):
    total_distance = 0
    depot = (0, 0)
    for route in solution:
        current_location = depot
        for customer_id in route:
            customer = customers[customer_id]
            total_distance += calculate_distance(current_location, (customer['X_Coordinate'], customer['Y_Coordinate']))
            current_location = (customer['X_Coordinate'], customer['Y_Coordinate'])
        total_distance += calculate_distance(current_location, depot)
    return total_distance

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
    fitness_history = []  # List to store the best fitness for each generation

    for generation in range(num_generations):
        fitness_values = [calculate_total_distance(sol, customers) for sol in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_values)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        population = new_population

        # Find the best solution and distance in the current generation
        best_solution = min(population, key=lambda sol: calculate_total_distance(sol, customers))
        best_distance = calculate_total_distance(best_solution, customers)

        # Store the best fitness value for this generation
        fitness_history.append(best_distance)

        # Streamlit progress
        st.write(f"Generation {generation + 1}: Best Distance = {best_distance}")

        # Check if the target fitness is achieved
        if best_distance <= target_fitness:
            st.success(f"Target fitness achieved in generation {generation + 1}!")
            return best_solution, best_distance, fitness_history

    # If target fitness was not achieved, return the best found solution after all generations
    st.warning("Target fitness not achieved within the generations.")
    return best_solution, best_distance, fitness_history

# Convert customer data into dictionary format
customers = {cust['Customer_ID']: cust for cust in st.session_state.customers}

# Button to run the Genetic Algorithm and show results
if st.button("Run Genetic Algorithm") and len(st.session_state.customers) >= 5:
    best_solution, best_distance, fitness_history = genetic_algorithm(customers, num_vehicles)
    st.write("Best Solution:", best_solution)
    st.write("Best Distance:", best_distance)

    # Plot the solution
    def plot_solution(solution, customers):
        plt.figure(figsize=(10, 8))

        # Plot depot
        plt.scatter(0, 0, color='red', s=100, marker='X', label="Depot")

        # Define colors for routes
        vehicle_colors = ['green', 'purple', 'orange', 'brown', 'cyan', 'blue', 'pink', 'yellow']

        # Create legend handles
        legend_handles = [mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label="Depot")]

        # Plot the routes for each vehicle
        for idx, route in enumerate(solution):
            color = vehicle_colors[idx % len(vehicle_colors)]
            current_location = (0, 0)  # Start from depot
            for customer_id in route:
                customer = customers[customer_id]
                plt.scatter(customer['X_Coordinate'], customer['Y_Coordinate'], color=color, s=50)
                plt.text(customer['X_Coordinate'], customer['Y_Coordinate'], f' {customer_id}', fontsize=9)
                plt.plot([current_location[0], customer['X_Coordinate']],
                         [current_location[1], customer['Y_Coordinate']], color=color, linestyle='-', marker='o')
                current_location = (customer['X_Coordinate'], customer['Y_Coordinate'])
            plt.plot([current_location[0], 0], [current_location[1], 0], color=color, linestyle='-', marker='X')

            # Add to legend
            legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='-', markersize=6, label=f"Vehicle {idx+1}"))

        plt.title("Vehicle Routing Problem - Solution Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(handles=legend_handles, loc="best")
        plt.grid(True)
        st.pyplot(plt)

    plot_solution(best_solution, customers)
