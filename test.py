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

default_data = {
    "Customer_ID": [1, 2, 3, 4, 5],
    "X_Coordinate": [10, 20, 30, 40, 50],
    "Y_Coordinate": [15, 25, 35, 45, 55],
    "Demand": [5, 10, 5, 10, 5]
}

# Allow user to edit customer data
data = pd.DataFrame(default_data)
st.write("Modify Customer Data Below:")
edited_data = st.experimental_data_editor(data, num_rows="dynamic")

# Define parameters
depot = (0, 0)

# Calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Fitness Function
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

# Initialize population
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

# Selection (Tournament Selection)
def select_parents(population, fitness_values):
    parents = []
    for _ in range(2):
        competitors = random.sample(range(len(population)), 5)
        best = min(competitors, key=lambda idx: fitness_values[idx])
        parents.append(population[best])
    return parents

# Crossover (Order Crossover)
def crossover(parent1, parent2):
    child = []
    for route1, route2 in zip(parent1, parent2):
        midpoint = len(route1) // 2
        child_route = route1[:midpoint] + [c for c in route2 if c not in route1[:midpoint]]
        child.append(child_route)
    return child

# Mutation (Swap Mutation)
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

# Create the customers dictionary from the DataFrame
customers = edited_data.set_index('Customer_ID').T.to_dict()

# Run the Genetic Algorithm
if st.button("Run Genetic Algorithm"):
    best_solution, best_distance, fitness_history = genetic_algorithm(customers, num_vehicles)
    st.write("Best Solution:", best_solution)
    st.write("Best Distance:", best_distance)

    # Plot the solution
    def plot_solution(solution, customers, depot):
        plt.figure(figsize=(10, 8))

        # Plot depot
        plt.scatter(depot[0], depot[1], color='red', s=100, marker='X', label="Depot")

        # Define colors for routes
        vehicle_colors = ['green', 'purple', 'orange', 'brown', 'cyan', 'blue', 'pink', 'yellow']

        # Create legend handles
        legend_handles = [mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label="Depot")]

        # Plot the routes for each vehicle
        for idx, route in enumerate(solution):
            color = vehicle_colors[idx % len(vehicle_colors)]
            current_location = depot
            for customer_id in route:
                customer = customers[customer_id]
                plt.scatter(customer['X_Coordinate'], customer['Y_Coordinate'], color=color, s=50)
                plt.text(customer['X_Coordinate'], customer['Y_Coordinate'], f' C{customer_id}', fontsize=9)
                plt.plot([current_location[0], customer['X_Coordinate']],
                         [current_location[1], customer['Y_Coordinate']], color=color, linestyle='-', marker='o')
                current_location = (customer['X_Coordinate'], customer['Y_Coordinate'])
            plt.plot([current_location[0], depot[0]], [current_location[1], depot[1]], color=color, linestyle='-', marker='X')

            # Add to legend
            legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='-', markersize=6, label=f"Vehicle {idx+1}"))

        plt.title("Vehicle Routing Problem - Solution Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(handles=legend_handles, loc="best")
        plt.grid(True)
        st.pyplot(plt)

    plot_solution(best_solution, customers, depot)
