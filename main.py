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

# =======================
# Reproducibility
# =======================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# =======================
# Gaussian Policy Network
# =======================
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)

    def forward(self, s):
        x = self.net(s)
        return self.mean(x), torch.exp(self.log_std(x))

    def sample_action(self, s):
        mean, std = self(s)
        a = torch.normal(mean, std)
        return np.clip(a.detach().numpy(), -1, 1)

# =======================
# Parameter Utilities
# =======================
def get_flat_params(model):
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])

def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(flat[idx:idx+n]).view_as(p))
        idx += n

# =======================
# Sparse Fitness Evaluation
# =======================
def evaluate_policy_sparse(policy, env,
                           episodes=3,
                           gamma=0.99,
                           max_steps=300,
                           sigma=0.02,
                           K=2):
    original = get_flat_params(policy).copy()

    def rollout():
        rewards = []
        for _ in range(episodes):
            s, _ = env.reset()
            done = False
            G, disc = 0.0, 1.0
            t = 0
            while not done and t < max_steps:
                a = policy.sample_action(torch.FloatTensor(s))
                s, r, done, _, _ = env.step(a)
                G += disc * r
                disc *= gamma
                t += 1
            rewards.append(G)
        return np.mean(rewards)

    total = 0.0
    for _ in range(K):
        noise = np.random.normal(0, sigma, len(original))
        set_flat_params(policy, original + noise)
        total += rollout()

    set_flat_params(policy, original)
    return total / K

def fitness_fn(individual, policy, env):
    set_flat_params(policy, individual)
    return evaluate_policy_sparse(policy, env)

# =======================
# Environment Factory
# =======================
def make_env(render=False):
    mode = "human" if render else None
    env = gym.make("MountainCarContinuous-v0", render_mode=mode)
    env.reset(seed=SEED)
    return env

# =======================
# Differential Evolution (WITH NORMALIZATION)
# =======================
def run_de(env, policy, dim, generations=25, pop_size=20):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(1)
    curve = []

    for g in range(generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        raw = np.array([fitness_fn(ind, policy, env) for ind in offspring])
        mean, std = raw.mean(), raw.std() + 1e-8
        norm = (raw - mean) / std

        for ind, f in zip(offspring, norm):
            ind.fitness.values = (f,)

        pop = toolbox.select(offspring, len(pop))
        hof.update(pop)
        curve.append(hof[0].fitness.values[0])
        print(f"DE Gen {g}: {curve[-1]:.2f}")

    return curve, hof[0]

# =======================
# Particle Swarm Optimization (WITH NORMALIZATION)
# =======================
def run_pso(env, policy, dim, generations=25, swarm_size=20):
    positions = np.random.uniform(-1, 1, (swarm_size, dim))
    velocities = np.zeros_like(positions)

    scores = np.array([fitness_fn(p, policy, env) for p in positions])
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    p_best = positions.copy()
    p_best_scores = scores.copy()

    g_idx = np.argmax(scores)
    g_best = positions[g_idx].copy()
    g_best_score = scores[g_idx]

    curve = [g_best_score]
    w, c1, c2 = 0.7, 1.4, 1.4

    for g in range(generations):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities = w * velocities + c1 * r1 * (p_best - positions) + c2 * r2 * (g_best - positions)
        positions = np.clip(positions + velocities, -1, 1)

        scores = np.array([fitness_fn(p, policy, env) for p in positions])
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)

        improved = scores > p_best_scores
        p_best[improved] = positions[improved]
        p_best_scores[improved] = scores[improved]

        idx = np.argmax(scores)
        if scores[idx] > g_best_score:
            g_best = positions[idx].copy()
            g_best_score = scores[idx]

        curve.append(g_best_score)
        print(f"PSO Gen {g}: {g_best_score:.2f}")

    return curve, g_best

# =======================
# Simulated Annealing (NO NORMALIZATION)
# =======================
def run_sa(env, policy, dim, generations=25):
    current = np.random.uniform(-1, 1, dim)
    current_score = fitness_fn(current, policy, env)

    best = current.copy()
    best_score = current_score
    curve = [best_score]

    T, alpha = 1.0, 0.95

    for g in range(generations):
        T *= alpha
        candidate = np.clip(current + np.random.normal(0, 0.1, dim), -1, 1)
        candidate_score = fitness_fn(candidate, policy, env)

        delta = candidate_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current, current_score = candidate, candidate_score

        if current_score > best_score:
            best, best_score = current.copy(), current_score

        curve.append(best_score)
        print(f"SA Gen {g}: {best_score:.2f}")

    return curve, best

# =======================
# PPO Baseline
# =======================
def run_ppo(env, steps=30000):
    model = PPO("MlpPolicy", env, seed=SEED, verbose=0)
    curve = []
    for _ in range(6):
        model.learn(5000)
        r, _ = evaluate_policy(model, env, n_eval_episodes=3)
        curve.append(r)
        print(f"PPO reward: {r:.2f}")
    return curve, model

# =======================
# Rendering
# =======================
def render_policy(env, policy=None, model=None, max_steps=500):
    s, _ = env.reset()
    for _ in range(max_steps):
        env.render()
        time.sleep(0.02)
        if model:
            a, _ = model.predict(s, deterministic=True)
        else:
            a = policy.sample_action(torch.FloatTensor(s))
        s, _, done, _, _ = env.step(a)
        if done:
            break
    env.close()

# =======================
# MAIN
# =======================
if __name__ == "__main__":

    env = make_env(render=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicy(state_dim, action_dim)
    dim = len(get_flat_params(policy))

    de_curve, de_best = run_de(env, policy, dim)
    pso_curve, pso_best = run_pso(env, policy, dim)
    sa_curve, sa_best = run_sa(env, policy, dim)
    ppo_curve, ppo_model = run_ppo(env)

    env.close()

    env_render = make_env(render=True)
    set_flat_params(policy, de_best)
    render_policy(env_render, policy=policy)

    env_render = make_env(render=True)
    set_flat_params(policy, pso_best)
    render_policy(env_render, policy=policy)

    env_render = make_env(render=True)
    set_flat_params(policy, sa_best)
    render_policy(env_render, policy=policy)

    env_render = make_env(render=True)
    render_policy(env_render, model=ppo_model)

    plt.figure(figsize=(10, 6))
    plt.plot(de_curve, label="DE")
    plt.plot(pso_curve, label="PSO")
    plt.plot(sa_curve, label="SA")
    plt.plot(ppo_curve, label="PPO")
    plt.xlabel("Generations / Updates")
    plt.ylabel("Normalized Fitness / Return")
    plt.legend()
    plt.grid(True)
    plt.show()
