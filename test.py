import streamlit as st
import pandas as pd
import numpy as np

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

# Initialize session state for customers data if not already initialized
if "customers" not in st.session_state:
    st.session_state.customers = []

# Function to add new customer data
def add_customer():
    # Assign customer ID
    customer_id = f"Customer {len(st.session_state.customers) + 1}"

    # Input fields for customer data
    x_coord = st.number_input(f"x-coordinate ({customer_id})", min_value=-100, max_value=100, step=1, key=f"x_coord_{customer_id}")
    y_coord = st.number_input(f"y-coordinate ({customer_id})", min_value=-100, max_value=100, step=1, key=f"y_coord_{customer_id}")
    demand = st.number_input(f"Demand ({customer_id})", min_value=1, step=1, key=f"demand_{customer_id}")
    
    # Store customer data if all fields are provided
    if x_coord and y_coord and demand:
        st.session_state.customers.append({
            "Customer_ID": customer_id,
            "X_Coordinate": x_coord,
            "Y_Coordinate": y_coord,
            "Demand": demand
        })

# Create a button to add new customer
if st.button("Add Customer"):
    add_customer()

# Display customer data as a table
if st.session_state.customers:
    customer_df = pd.DataFrame(st.session_state.customers)
    st.write("Customer Data Table:")
    st.dataframe(customer_df)

# Option to clear all customer data
if st.button("Clear All Customers"):
    st.session_state.customers = []
    st.success("All customer data cleared.")

# Genetic Algorithm related code
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
def initialize_population(customers, num_vehicles, population_size):
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
def mutate(solution, mutation_rate):
    for route in solution:
        if random.random() < mutation_rate and len(route) > 1:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
    return solution

# Genetic Algorithm
def genetic_algorithm(customers, num_vehicles, num_generations, population_size, mutation_rate):
    population = initialize_population(customers, num_vehicles, population_size)

    best_solution = None
    best_distance = float('inf')
    fitness_history = []  # List to store the best fitness for each generation

    for generation in range(num_generations):
        fitness_values = [calculate_total_distance(sol, customers) for sol in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_values)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent2, parent1), mutation_rate)
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

# Run the Genetic Algorithm
if st.button("Run Genetic Algorithm") and len(st.session_state.customers) >= 5:
    customers = {cust['Customer_ID']: cust for cust in st.session_state.customers}

    best_solution, best_distance, fitness_history = genetic_algorithm(customers, num_vehicles, num_generations, population_size, mutation_rate)
    
    st.write("Best Solution:", best_solution)
    st.write("Best Distance:", best_distance)

    # Plot the solution (You can reuse the previous plot_solution function for visualization)
    # ...
