import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- GA Configuration ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
# Probability of flipping a bit in a chromosome
MUTATION_RATE = 0.02
# Size of the tournament for parent selection
TOURNAMENT_SIZE = 5
# Penalty for model complexity. A small value encourages fewer features.
COMPLEXITY_PENALTY_FACTOR = 0.001

# --- Data Loading ---
# We load data globally so the fitness function can access it without passing it around.
# This is a common pattern in scripts like this for simplicity.
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)
NUM_FEATURES = X.shape[1]

# --- Core GA Functions ---


def create_initial_population():
    """Creates a list of random chromosomes to start the evolution."""
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = np.random.randint(0, 2, NUM_FEATURES).tolist()
        population.append(chromosome)
    return population


def get_selected_features(chromosome):
    """Decodes a chromosome into a list of feature names."""
    return [X.columns[i] for i, gene in enumerate(chromosome) if gene == 1]


def calculate_fitness(chromosome):
    """
    Calculates the fitness of a chromosome.
    Fitness = Accuracy - (Number of features * Penalty)
    """
    selected_features = get_selected_features(chromosome)

    # If a chromosome has no features, its fitness is 0. Avoids errors.
    if not selected_features:
        return 0.0

    X_subset = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Apply penalty for complexity (number of features)
    num_selected = len(selected_features)
    fitness = accuracy - (num_selected * COMPLEXITY_PENALTY_FACTOR)

    return fitness


def selection(population, fitness_scores):
    """
    Selects two parents using tournament selection.
    In a tournament, k individuals are chosen randomly, and the fittest becomes a parent.
    """
    parents = []
    for _ in range(2):  # Select two parents
        tournament_indices = random.sample(range(POPULATION_SIZE), TOURNAMENT_SIZE)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index_in_tournament = np.argmax(tournament_fitness)
        winner_index_in_population = tournament_indices[winner_index_in_tournament]
        parents.append(population[winner_index_in_population])
    return parents[0], parents[1]


def crossover(parent1, parent2):
    """
    Performs single-point crossover to create a child.
    A random point is chosen, and the child gets genes from parent1 before this point
    and genes from parent2 after this point.
    """
    crossover_point = random.randint(1, NUM_FEATURES - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutation(chromosome):
    """
    Applies mutation to a chromosome.
    For each gene, there's a small chance (MUTATION_RATE) it will be flipped.
    """
    mutated_chromosome = []
    for gene in chromosome:
        if random.random() < MUTATION_RATE:
            mutated_chromosome.append(1 - gene)  # Flip the bit
        else:
            mutated_chromosome.append(gene)
    return mutated_chromosome


# --- Main GA Execution ---


def run_genetic_algorithm():
    """The main orchestration logic for the GA."""
    population = create_initial_population()
    best_chromosome_overall = None
    best_fitness_overall = -1.0
    print("--- Starting Genetic Algorithm ---")

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [calculate_fitness(chromo) for chromo in population]

        # Find the best of the current generation
        best_fitness_gen = max(fitness_scores)
        best_chromosome_gen = population[np.argmax(fitness_scores)]

        # Update overall best if current generation is better
        if best_fitness_gen > best_fitness_overall:
            best_fitness_overall = best_fitness_gen
            best_chromosome_overall = best_chromosome_gen

        print(
            f"Generation {generation + 1}/{NUM_GENERATIONS} | "
            f"Best Fitness: {best_fitness_overall:.4f} | "
            f"Features: {sum(best_chromosome_overall)}"
        )

        # Create the next generation
        next_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = selection(population, fitness_scores)

            # Create two children
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)

            next_population.append(mutation(child1))
            next_population.append(mutation(child2))

        population = next_population

    print("--- Genetic Algorithm Finished ---")
    return best_chromosome_overall
    # --- Missing Functions (added manually) ---

def calculate_fitness(chromosome):
    """Evaluates a chromosome by training a Logistic Regression model on the selected features."""
    selected_features = [i for i in range(NUM_FEATURES) if chromosome[i] == 1]
    if len(selected_features) == 0:
        return 0  # Avoid empty feature sets

    X_selected = X.iloc[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Encourage smaller feature sets (penalty)
    penalty = len(selected_features) * 0.001
    return acc - penalty


def crossover(parent1, parent2):
    """Single-point crossover between two parents."""
    point = random.randint(1, NUM_FEATURES - 1)
    child = parent1[:point] + parent2[point:]
    return child


def mutation(chromosome, mutation_rate=0.05):
    """Randomly flips bits in the chromosome."""
    for i in range(NUM_FEATURES):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


if __name__ == "__main__":
    final_best_chromosome = run_genetic_algorithm()

    # Final evaluation of the best chromosome found
    selected_features = get_selected_features(final_best_chromosome)
    final_fitness = calculate_fitness(final_best_chromosome)

    # To get the raw accuracy without the penalty, we can recalculate it
    X_subset = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, predictions)

    print("\n--- Optimal Feature Set Found ---")
    print(
        f"Number of features selected: {len(selected_features)} out of {NUM_FEATURES}"
    )
    print(f"Final Model Accuracy: {final_accuracy:.4f}")
    print("\nSelected Features:")
    for feature in selected_features:
        print(f"- {feature}")
