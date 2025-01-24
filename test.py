import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Title
st.title("Vehicle Routing Problem (VRP) Using Genetic Algorithm")
st.write("Enter up to 20 customers with their coordinates (x, y) in range 0 - 100.")

# Define empty lists to store customer data
customer_coords = {}

# Create a table-like layout for input
col1, col2, col3 = st.columns([1, 1, 1])  # Define three columns

# Collect user input for each customer
for i in range(1, 21):
    # Label for customer
    customer_name = f"Customer {i}"

    # Input fields for x and y coordinates
    x_coord = col2.number_input(f"x-coordinate ({customer_name})", min_value=0, max_value=100, step=1, key=f"x_coord_{i}")
    y_coord = col3.number_input(f"y-coordinate ({customer_name})", min_value=0, max_value=100, step=1, key=f"y_coord_{i}")

    # Store data if coordinates are provided
    if x_coord and y_coord:
        customer_coords[customer_name] = (x_coord, y_coord)

# GA Parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200
n_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=10, value=3, step=1)
vehicle_capacity = st.number_input("Vehicle Capacity", min_value=1, max_value=100, value=50, step=1)

# Define color palette for customer markers
colors = sns.color_palette("pastel", len(customer_coords))

# "Submit" button to run the algorithm
if st.button("Submit"):
    # Plot initial customer locations with connections
    fig, ax = plt.subplots()
    for i, (customer, (customer_x, customer_y)) in enumerate(customer_coords.items()):
        color = colors[i]
        ax.scatter(customer_x, customer_y, c=[color], s=1200, zorder=2)

        # Display customer label on the plot
        ax.annotate(customer, (customer_x, customer_y), fontsize=12, ha='center', va='bottom')

    # Draw faint lines between each pair of customers
    for i, (cust1, (x1, y1)) in enumerate(customer_coords.items()):
        for j, (cust2, (x2, y2)) in enumerate(customer_coords.items()):
            if i != j:
                ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Define helper functions

    def dist_two_customers(cust1, cust2):
        coord1 = customer_coords[cust1]
        coord2 = customer_coords[cust2]
        return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def total_distance(route):
        total_dist = 0
        for i in range(len(route) - 1):
            total_dist += dist_two_customers(route[i], route[i + 1])
        total_dist += dist_two_customers(route[-1], route[0])  # Return to depot
        return total_dist

    def initial_population(customers, n_population):
        all_routes = list(permutations(customers))
        shuffle(all_routes)
        return [list(route) for route in all_routes[:n_population]]

    def fitness(population):
        distances = [total_distance(route) for route in population]
        max_distance = max(distances)
        fitness_scores = [max_distance - d for d in distances]
        total_fitness = sum(fitness_scores)
        return [f / total_fitness for f in fitness_scores]

    def select_parents(population, fitness_scores):
        cumulative_probs = np.cumsum(fitness_scores)
        selected = []
        for _ in range(len(population)):
            rand = random.random()
            for i, prob in enumerate(cumulative_probs):
                if rand <= prob:
                    selected.append(population[i])
                    break
        return selected

    def crossover(parent1, parent2):
        cut = random.randint(1, len(parent1) - 2)
        child1 = parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]
        child2 = parent2[:cut] + [c for c in parent1 if c not in parent2[:cut]]
        return child1, child2

    def mutate(route):
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    # Run GA
    customers = list(customer_coords.keys())
    population = initial_population(customers, n_population)

    for generation in range(n_generations):
        fitness_scores = fitness(population)
        parents = select_parents(population, fitness_scores)
        next_population = []

        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            if random.random() < mutation_per:
                child1 = mutate(child1)
            if random.random() < mutation_per:
                child2 = mutate(child2)
            next_population.extend([child1, child2])

        population = next_population[:n_population]

    best_route = min(population, key=total_distance)
    min_distance = total_distance(best_route)

    # Display results
    st.write("Best Route:", best_route)
    st.write("Minimum Distance:", min_distance)

    # Plot best route
    x_coords = [customer_coords[c][0] for c in best_route + [best_route[0]]]
    y_coords = [customer_coords[c][1] for c in best_route + [best_route[0]]]

    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, '-o', label='Best Route')
    plt.title("Best Route Found")
    plt.legend()
    fig.set_size_inches(16, 12)
    st.pyplot(fig)
