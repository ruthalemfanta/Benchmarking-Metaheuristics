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
import time

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- Sparse Variant Wrapper (optional - to make rewards sparser) ---
class SparseLunarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Make sparser: remove dense shaping (position, velocity, angle, leg contact bonuses)
        # Keep only landing/crash + engine cost
        engine_cost = -0.3 if action == 1 or action == 3 else 0  # Main engine or side
        side_cost = -0.03 if action == 2 or action == 4 else 0  # Side engines (approx)
        reward = engine_cost + side_cost
        if terminated:
            if self.env.lander.linearVelocity.length > 10 or self.env.lander.angularVelocity > 20:  # Crash check approx
                reward -= 100
            else:
                reward += 100  # Landed
        return obs, reward, terminated, truncated, info

# --- Module 1: Policy Network (Discrete for LunarLander-v2) ---
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.head = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.net(state)
        logits = self.head(x)
        return logits

    def sample_action(self, state):
        logits = self(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

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
def evaluate_policy_custom(policy, env, episodes=10, gamma=0.99, max_steps=1000):
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
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
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
def create_env(render=False):
    mode = "human" if render else None
    env = gym.make('LunarLander-v3', render_mode=mode)
    env.reset(seed=SEED)
    # For sparse variant, uncomment:
    # env = SparseLunarWrapper(env)
    return env

# --- Module 4: Optimizers ---
# 4.1 Differential Evolution (DE) with DEAP
def run_de(env, policy, param_size, generations=100, pop_size=50):
    # Clear existing classes if they exist to avoid conflicts
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)  # pylint: disable=no-member

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=param_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", deap_fitness, policy=policy, env=env)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    logbook = tools.Logbook()

    for gen in range(generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(f"DE Gen {gen}: Max Fitness {record['max']}")

    return logbook.select("max"), hof[0]

# 4.2 Particle Swarm Optimization (PSO) custom
def run_pso(env, policy, param_size, generations=100, swarm_size=50):
    bounds = [-1, 1]
    positions = np.random.uniform(bounds[0], bounds[1], (swarm_size, param_size))
    velocities = np.random.uniform(-0.1, 0.1, (swarm_size, param_size))
    personal_best_pos = positions.copy()
    personal_best_scores = np.array([deap_fitness(pos, policy, env)[0] for pos in positions])
    global_best_pos = personal_best_pos[np.argmax(personal_best_scores)]
    global_best_score = np.max(personal_best_scores)

    w, c1, c2 = 0.8, 1.5, 1.5
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

    return convergence, global_best_pos

# 4.3 Simulated Annealing (SA) custom
def run_sa(env, policy, param_size, generations=100):
    current_params = np.random.uniform(-1, 1, param_size)
    current_score = deap_fitness(current_params, policy, env)[0]
    best_params = current_params.copy()
    best_score = current_score

    T = 1.0
    T_min = 0.01
    alpha = 0.95
    convergence = [best_score]

    for gen in range(1, generations):
        T = max(T * alpha, T_min)
        neighbor = current_params + np.random.normal(0, 0.2, param_size)
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

    return convergence, best_params

# 4.4 PPO Baseline
def run_ppo(env, timesteps=90000, eval_episodes=10):
    model = PPO("MlpPolicy", env, verbose=0, seed=SEED, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)
    convergence = []
    for i in range(0, timesteps, 10000):
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
        convergence.append(mean_reward)
        print(f"PPO Step {i+10000}: Mean Reward {mean_reward}")
    return convergence, model

# --- Visualization ---
def render_best(env_render, policy_or_model, is_ppo=False, max_steps=1000):
    state, _ = env_render.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        env_render.render()
        time.sleep(0.01)  # Slow for viewing
        if is_ppo:
            action, _ = policy_or_model.predict(state, deterministic=True)
        else:
            state_tensor = torch.FloatTensor(state)
            action = policy_or_model.sample_action(state_tensor)
        state, _, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        step += 1
    env_render.close()

# --- Main ---
if __name__ == "__main__":
    env = create_env(render=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # 4
    policy = DiscretePolicy(state_dim, action_dim)
    param_size = len(get_flat_params(policy))

    print("Running DE...")
    de_convergence, de_best_ind = run_de(env, policy, param_size)

    print("Running PSO...")
    pso_convergence, pso_best_pos = run_pso(env, policy, param_size)

    print("Running SA...")
    sa_convergence, sa_best_params = run_sa(env, policy, param_size)

    print("Running PPO...")
    ppo_convergence, ppo_model = run_ppo(env)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(de_convergence, label='DE')
    plt.plot(pso_convergence, label='PSO')
    plt.plot(sa_convergence, label='SA')
    plt.plot(ppo_convergence, label='PPO')
    plt.xlabel('Generations / Updates')
    plt.ylabel('Average Reward (Fitness)')
    plt.title('Convergence Curves: Neuroevolution vs PPO on LunarLander-v3')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence.png')
    plt.show()

    print("Final Rewards: DE", de_convergence[-1], "PSO", pso_convergence[-1], "SA", sa_convergence[-1], "PPO", ppo_convergence[-1])

    # Render best
    env_render = create_env(render=True)

    print("Rendering best DE...")
    set_flat_params(policy, de_best_ind)
    render_best(env_render, policy, is_ppo=False)

    print("Rendering best PSO...")
    set_flat_params(policy, pso_best_pos)
    render_best(env_render, policy, is_ppo=False)

    print("Rendering best SA...")
    set_flat_params(policy, sa_best_params)
    render_best(env_render, policy, is_ppo=False)

    print("Rendering PPO...")
    render_best(env_render, ppo_model, is_ppo=True)

    env.close()