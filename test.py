import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Title
st.title("Vehicle Routing Problem (VRP) Solver with GA")
st.write("Input the depot and customer coordinates to solve the VRP using a Genetic Algorithm.")

# Depot Input
st.header("Depot Coordinates")
depot_x = st.number_input("Depot X-coordinate", min_value=0, max_value=100, step=1, key="depot_x")
depot_y = st.number_input("Depot Y-coordinate", min_value=0, max_value=100, step=1, key="depot_y")

# Customer Input
st.header("Customer Coordinates")
num_customers = st.slider("Number of Customers", min_value=1, max_value=20, step=1, key="num_customers")
customer_coords = {}

for i in range(1, num_customers + 1):
    x = st.number_input(f"Customer {i} X-coordinate", min_value=0, max_value=100, step=1, key=f"cust_{i}_x")
    y = st.number_input(f"Customer {i} Y-coordinate", min_value=0, max_value=100, step=1, key=f"cust_{i}_y")
    customer_coords[f"Customer {i}"] = (x, y)

# GA Parameters
st.header("Genetic Algorithm Parameters")
n_population = st.number_input("Population Size", min_value=10, max_value=1000, step=10, value=250)
crossover_per = st.slider("Crossover Percentage", min_value=0.1, max_value=1.0, step=0.1, value=0.8)
mutation_per = st.slider("Mutation Percentage", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
n_generations = st.number_input("Number of Generations", min_value=10, max_value=1000, step=10, value=200)

# Visualize Input Data
if st.button("Visualize Data"):
    fig, ax = plt.subplots()

    # Plot depot
    ax.scatter(depot_x, depot_y, c='red', s=200, label='Depot', zorder=2)
    ax.annotate("Depot", (depot_x, depot_y), fontsize=12, ha='center', va='bottom')

    # Plot customers
    colors = sns.color_palette("pastel", num_customers)
    for i, (customer, (x, y)) in enumerate(customer_coords.items()):
        color = colors[i]
        ax.scatter(x, y, c=[color], s=120, zorder=2)
        ax.annotate(customer, (x, y), fontsize=10, ha='center', va='bottom')

    ax.set_title("Depot and Customer Locations")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.grid(True)
    st.pyplot(fig)

# Distance Calculation
def dist_two_points(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Total Distance for a Route
def total_distance(route, depot):
    total_dist = dist_two_points(depot, route[0])  # Start from depot
    for i in range(len(route) - 1):
        total_dist += dist_two_points(route[i], route[i + 1])
    total_dist += dist_two_points(route[-1], depot)  # Return to depot
    return total_dist

# Genetic Algorithm Functions
def initial_population(customers, n_population):
    return [random.sample(customers, len(customers)) for _ in range(n_population)]

def fitness(population, depot):
    distances = [total_distance(route, depot) for route in population]
    fitness_scores = [1 / dist for dist in distances]
    return fitness_scores, distances

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probs = [score / total_fitness for score in fitness_scores]
    return random.choices(population, weights=probs, k=2)

def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]
    child2 = parent2[:cut] + [c for c in parent1 if c not in parent2[:cut]]
    return child1, child2

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(customers, depot, n_population, n_generations, crossover_rate, mutation_rate):
    population = initial_population(customers, n_population)
    best_route = None
    best_distance = float('inf')

    for _ in range(n_generations):
        fitness_scores, distances = fitness(population, depot)
        new_population = []

        for _ in range(n_population // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population
        min_distance = min(distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_route = population[distances.index(min_distance)]

    return best_route, best_distance

# Run GA
if st.button("Run Genetic Algorithm"):
    depot_coords = (depot_x, depot_y)
    customer_list = list(customer_coords.values())

    best_route, best_distance = genetic_algorithm(
        customer_list, depot_coords, n_population, n_generations, crossover_per, mutation_per
    )

    # Visualize Result
    fig, ax = plt.subplots()

    # Plot depot
    ax.scatter(depot_x, depot_y, c='red', s=200, label='Depot', zorder=2)
    ax.annotate("Depot", (depot_x, depot_y), fontsize=12, ha='center', va='bottom')

    # Plot best route
    route_x = [depot_x] + [coord[0] for coord in best_route] + [depot_x]
    route_y = [depot_y] + [coord[1] for coord in best_route] + [depot_y]
    ax.plot(route_x, route_y, '--o', label='Best Route', zorder=1)

    # Plot customers
    colors = sns.color_palette("pastel", len(customer_list))
    for i, (x, y) in enumerate(customer_list):
        color = colors[i]
        ax.scatter(x, y, c=[color], s=120, zorder=2)
        ax.annotate(f"Customer {i+1}", (x, y), fontsize=10, ha='center', va='bottom')

    ax.set_title("Optimized Route")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.grid(True)
    st.pyplot(fig)

    st.write(f"**Best Route Distance:** {best_distance:.2f}")
