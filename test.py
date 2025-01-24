import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Title and Description
st.title("Traveling Salesman Problem with GA")
st.write("Visualize and solve TSP using Genetic Algorithm. Enter up to 10 cities with coordinates (x, y) within the range 0 - 100.")

# Define city names with icons
city_icons = {
    "Johor": "♕",
    "Melaka": "♖",
    "Negeri Sembilan": "♗",
    "Selangor": "♘",
    "Kuala Lumpur": "♙",
    "Perak": "♔",
    "Kedah": "♚",
    "Kelantan": "♛",
    "Terengganu": "♜",
    "Pulau Pinang": "♝"
}

# Define empty lists to store city names and coordinates
city_coords = {}

# Layout for input
st.sidebar.header("City Input")
for i in range(1, 11):
    city_name = st.sidebar.selectbox(f"City {i} Name", options=[""] + list(city_icons.keys()), key=f"city_name_{i}")
    x_coord = st.sidebar.number_input(f"City {i} X-coordinate", min_value=0, max_value=100, key=f"x_coord_{i}")
    y_coord = st.sidebar.number_input(f"City {i} Y-coordinate", min_value=0, max_value=100, key=f"y_coord_{i}")
    
    if city_name:
        city_coords[city_name] = (x_coord, y_coord)

# Genetic Algorithm Parameters
n_population = 250
n_generations = 200
crossover_per = 0.8
mutation_per = 0.2

# Generate City Plot
if st.sidebar.button("Visualize Cities"):
    colors = sns.color_palette("pastel", len(city_coords))
    fig, ax = plt.subplots()
    
    for i, (city, (x, y)) in enumerate(city_coords.items()):
        ax.scatter(x, y, color=colors[i], s=300, label=city, zorder=2)
        ax.annotate(f"{city_icons.get(city, '')} {city}", (x, y), fontsize=12, ha='center', va='bottom')
    
    ax.grid(color="lightgray", linestyle="dotted")
    ax.set_title("City Locations", fontsize=16)
    plt.legend(loc="upper right")
    fig.set_size_inches(10, 6)
    st.pyplot(fig)

# Genetic Algorithm Functions
def initial_population(cities, n_population=250):
    return [list(ind) for ind in random.sample(list(permutations(cities)), n_population)]

def dist_two_cities(city1, city2):
    coords1, coords2 = city_coords[city1], city_coords[city2]
    return np.sqrt((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2)

def total_dist(route):
    return sum(dist_two_cities(route[i], route[(i + 1) % len(route)]) for i in range(len(route)))

def fitness(population):
    distances = np.array([total_dist(ind) for ind in population])
    fitness_scores = 1 / (1 + distances)
    return fitness_scores / fitness_scores.sum()

def roulette_selection(population, fitness_scores):
    cumsum = np.cumsum(fitness_scores)
    rand_val = random.random()
    return population[np.searchsorted(cumsum, rand_val)]

def crossover(parent1, parent2):
    size = len(parent1)
    cut = random.randint(0, size - 1)
    child1 = parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]
    child2 = parent2[:cut] + [c for c in parent1 if c not in parent2[:cut]]
    return child1, child2

def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def genetic_algorithm(cities, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities, n_population)
    for _ in range(n_generations):
        fitness_scores = fitness(population)
        new_population = []
        while len(new_population) < n_population:
            parent1 = roulette_selection(population, fitness_scores)
            parent2 = roulette_selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < mutation_per:
                child1 = mutate(child1)
            if random.random() < mutation_per:
                child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:n_population]
    return min(population, key=total_dist)

# Run GA and Visualize
if st.sidebar.button("Run Genetic Algorithm"):
    if len(city_coords) < 2:
        st.warning("Please input at least 2 cities to proceed.")
    else:
        best_route = genetic_algorithm(list(city_coords.keys()), n_population, n_generations, crossover_per, mutation_per)
        shortest_distance = total_dist(best_route)
        
        st.subheader("Optimal Route")
        st.write(f"Shortest Distance: {shortest_distance:.2f}")
        st.write(" -> ".join(best_route))
        
        # Plot Best Route
        x_coords, y_coords = zip(*[city_coords[city] for city in best_route + [best_route[0]]])
        fig, ax = plt.subplots()
        ax.plot(x_coords, y_coords, 'o-', label='Optimal Path')
        for i, city in enumerate(best_route):
            ax.annotate(f"{city_icons[city]} {city}", (x_coords[i], y_coords[i]), fontsize=12, ha='center')
        
        ax.grid(color="lightgray", linestyle="dotted")
        ax.set_title("Optimal Route Visualization", fontsize=16)
        plt.legend()
        fig.set_size_inches(10, 6)
        st.pyplot(fig)
