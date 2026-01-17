import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import random
import os
from scipy.stats import ranksums


# DEAP is already installed via requirements.txt

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- Module 1: Policy Network (Gaussian for continuous actions) ---
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(32, action_dim)
        self.log_std_head = nn.Linear(32, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        std = torch.exp(log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self(state)
        action = torch.normal(mean, std)
        return np.clip(action.detach().numpy(), -1, 1)  # Clip to action bounds

# Flatten/unflatten params for metaheuristics
def get_flat_params(model):
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        param_len = p.numel()
        p.data.copy_(torch.tensor(flat_params[idx:idx + param_len]).view_as(p))
        idx += param_len

# --- Module 2: Fitness Evaluation ---
def evaluate_policy_custom(policy, env, episodes=10, gamma=0.99, max_steps=200):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        discount = 1.0
        while not done and step < max_steps:
            state_tensor = torch.FloatTensor(state)
            action = policy.sample_action(state_tensor)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward * discount
            discount *= gamma
            step += 1
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# DEAP wrapper for fitness
def deap_fitness(individual, policy, env):
    set_flat_params(policy, individual)
    return (evaluate_policy_custom(policy, env),)

# --- Module 3: Environment Setup ---
def create_env():
    env = gym.make('MountainCarContinuous-v0')
    env.reset(seed=SEED)
    return env

# --- Module 4: Optimizers ---
# 4.1 Differential Evolution (DE) with DEAP
def run_de(env, policy, param_size, generations=30, pop_size=20):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=param_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", deap_fitness, policy=policy, env=env)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    for gen in range(generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, len(pop))
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(f"DE Gen {gen}: Max Fitness {record['max']}")

    return logbook.select("max")

# 4.2 Particle Swarm Optimization (PSO) with DEAP (using simple PSO variant)
def run_pso(env, policy, param_size, generations=30, swarm_size=20):
    # DEAP doesn't have built-in PSO, so use a custom simple implementation with numpy
    # (Since DEAP might not be full for PSO, fallback to numpy for modularity)
    bounds = [-1, 1]  # Param bounds
    positions = np.random.uniform(bounds[0], bounds[1], (swarm_size, param_size))
    velocities = np.random.uniform(-0.1, 0.1, (swarm_size, param_size))
    personal_best_pos = positions.copy()
    personal_best_scores = np.array([deap_fitness(pos, policy, env)[0] for pos in positions])
    global_best_pos = personal_best_pos[np.argmax(personal_best_scores)]
    global_best_score = np.max(personal_best_scores)

    w, c1, c2 = 0.8, 1.5, 1.5  # PSO params
    convergence = [global_best_score]

    for gen in range(1, generations):
        r1 = np.random.rand(swarm_size, param_size)
        r2 = np.random.rand(swarm_size, param_size)
        velocities = w * velocities + c1 * r1 * (personal_best_pos - positions) + c2 * r2 * (global_best_pos - positions)
        positions += velocities
        positions = np.clip(positions, bounds[0], bounds[1])

        scores = np.array([deap_fitness(pos, policy, env)[0] for pos in positions])
        improved = scores > personal_best_scores
        personal_best_pos[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]

        if np.max(scores) > global_best_score:
            global_best_score = np.max(scores)
            global_best_pos = positions[np.argmax(scores)]

        convergence.append(global_best_score)
        print(f"PSO Gen {gen}: Max Fitness {global_best_score}")

    return convergence

# 4.3 Simulated Annealing (SA) with custom numpy
def run_sa(env, policy, param_size, generations=30):
    # Simple SA implementation
    current_params = np.random.uniform(-1, 1, param_size)
    current_score = deap_fitness(current_params, policy, env)[0]
    best_params = current_params.copy()
    best_score = current_score

    T = 1.0  # Initial temp
    T_min = 0.01
    alpha = 0.95
    convergence = [best_score]

    for gen in range(1, generations):
        T = max(T * alpha, T_min)
        neighbor = current_params + np.random.normal(0, 0.1, param_size)
        neighbor = np.clip(neighbor, -1, 1)
        neighbor_score = deap_fitness(neighbor, policy, env)[0]

        delta = neighbor_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_params = neighbor
            current_score = neighbor_score

        if current_score > best_score:
            best_score = current_score
            best_params = current_params

        convergence.append(best_score)
        print(f"SA Gen {gen}: Best Fitness {best_score}")

    return convergence

# 4.4 PPO Baseline
def run_ppo(env, timesteps=20000, eval_episodes=10):
    model = PPO("MlpPolicy", env, verbose=0, seed=SEED, learning_rate=3e-4, n_steps=512, batch_size=32)
    convergence = []
    for i in range(0, timesteps, 2000):  # Log every 2000 steps to simulate "generations"
        model.learn(total_timesteps=2000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
        convergence.append(mean_reward)
        print(f"PPO Step {i+2000}: Mean Reward {mean_reward}")
    return convergence

# --- Main: Run all and Plot ---
if __name__ == "__main__":
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = GaussianPolicy(state_dim, action_dim)
    param_size = len(get_flat_params(policy))

    print("Running DE...")
    de_convergence = run_de(env, policy, param_size)

    print("Running PSO...")
    pso_convergence = run_pso(env, policy, param_size)

    print("Running SA...")
    sa_convergence = run_sa(env, policy, param_size)

    print("Running PPO...")
    ppo_convergence = run_ppo(env)

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(de_convergence, label='DE')
    plt.plot(pso_convergence, label='PSO')
    plt.plot(sa_convergence, label='SA')
    plt.plot(ppo_convergence, label='PPO')
    plt.xlabel('Generations / Updates')
    plt.ylabel('Average Reward (Fitness)')
    plt.title('Convergence Curves: Neuroevolution vs PPO')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence.png')
    plt.show()

    # For stats (Wilcoxon in paper: use scipy.stats.wilcoxon on final rewards)
    # --- Wilcoxon Rank-Sum Tests (Statistical Significance) ---
    print("\nWilcoxon Rank-Sum Tests (Metaheuristics vs PPO):")

    stat_de, p_de = ranksums(de_convergence, ppo_convergence)
    stat_pso, p_pso = ranksums(pso_convergence, ppo_convergence)
    stat_sa, p_sa = ranksums(sa_convergence, ppo_convergence)

    print(f"DE vs PPO:  statistic={stat_de:.4f}, p-value={p_de:.4e}")
    print(f"PSO vs PPO: statistic={stat_pso:.4f}, p-value={p_pso:.4e}")
    print(f"SA vs PPO:  statistic={stat_sa:.4f}, p-value={p_sa:.4e}")

    if p_de < 0.05:
        print("DE vs PPO: statistically significant difference (p < 0.05)")
    else:
        print("DE vs PPO: no statistically significant difference")

    if p_pso < 0.05:
        print("PSO vs PPO: statistically significant difference (p < 0.05)")
    else:
        print("PSO vs PPO: no statistically significant difference")

    if p_sa < 0.05:
        print("SA vs PPO: statistically significant difference (p < 0.05)")
    else:
        print("SA vs PPO: no statistically significant difference")
