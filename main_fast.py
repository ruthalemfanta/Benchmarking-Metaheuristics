import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from scipy.stats import ranksums

# =============================
# 1. Reproducibility
# =============================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# =============================
# 2. Sparse Reward Wrapper
# =============================
class SparseMountainCar(gym.Wrapper):
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        position = obs[0]

        reward = 0.0
        if terminated and position >= 0.45:
            reward = 100.0
        elif terminated:
            reward = -1.0

        return obs, reward, terminated, truncated, info


def make_env():
    env = gym.make("MountainCarContinuous-v0")
    env = SparseMountainCar(env)
    env.reset(seed=SEED)
    return env

# =============================
# 3. Policy Network
# =============================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),  # Smaller network
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.mean = nn.Linear(16, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.net(x)
        return self.mean(x), torch.exp(self.log_std)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self(state)
        action = torch.normal(mean, std)
        return np.clip(action.detach().numpy(), -1, 1)

# =============================
# 4. Utilities
# =============================
def flatten_params(model):
    return np.concatenate([p.data.numpy().flatten() for p in model.parameters()])


def set_params(model, flat):
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(torch.tensor(flat[idx:idx+size]).view_as(p))
        idx += size


def evaluate_policy_custom(params, model, episodes=5):  # Reduced from 30 to 5
    set_params(model, params)
    rewards = []

    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        done = False
        total = 0
        steps = 0

        while not done and steps < 200:  # Max 200 steps per episode
            action = model.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1

        env.close()
        rewards.append(total)

    return np.mean(rewards)

# =============================
# 5. Differential Evolution
# =============================
def run_de(model, dim, generations=10, pop_size=10):  # Reduced parameters
    pop = np.random.uniform(-1, 1, (pop_size, dim))
    fitness = np.array([evaluate_policy_custom(p, model) for p in pop])
    history = []

    for g in range(generations):
        for i in range(pop_size):
            a, b, c = pop[np.random.choice(pop_size, 3, replace=False)]
            trial = np.clip(a + 0.8 * (b - c), -1, 1)
            score = evaluate_policy_custom(trial, model)
            if score > fitness[i]:
                pop[i] = trial
                fitness[i] = score

        history.append(np.max(fitness))
        print(f"DE Gen {g}: {history[-1]:.2f}")

    return history, np.max(fitness)

# =============================
# 6. Particle Swarm Optimization
# =============================
def run_pso(model, dim, generations=10, swarm=10):  # Reduced parameters
    pos = np.random.uniform(-1, 1, (swarm, dim))
    vel = np.zeros_like(pos)

    pbest = pos.copy()
    pbest_score = np.array([evaluate_policy_custom(p, model) for p in pos])
    gbest = pbest[np.argmax(pbest_score)]
    gbest_score = np.max(pbest_score)

    history = []

    for g in range(generations):
        for i in range(swarm):
            vel[i] = (
                0.7 * vel[i]
                + 1.5 * random.random() * (pbest[i] - pos[i])
                + 1.5 * random.random() * (gbest - pos[i])
            )
            pos[i] = np.clip(pos[i] + vel[i], -1, 1)

            score = evaluate_policy_custom(pos[i], model)
            if score > pbest_score[i]:
                pbest[i] = pos[i]
                pbest_score[i] = score

        gbest_score = np.max(pbest_score)
        gbest = pbest[np.argmax(pbest_score)]
        history.append(gbest_score)
        print(f"PSO Gen {g}: {gbest_score:.2f}")

    return history, gbest_score

# =============================
# 7. Simulated Annealing
# =============================
def run_sa(model, dim, generations=10):  # Reduced parameters
    current = np.random.uniform(-1, 1, dim)
    current_score = evaluate_policy_custom(current, model)
    best_score = current_score
    history = []

    T = 1.0
    for g in range(generations):
        T *= 0.95
        neighbor = np.clip(current + np.random.normal(0, 0.1, dim), -1, 1)
        score = evaluate_policy_custom(neighbor, model)

        if score > current_score or random.random() < np.exp((score - current_score) / T):
            current, current_score = neighbor, score

        best_score = max(best_score, current_score)
        history.append(best_score)
        print(f"SA Gen {g}: {best_score:.2f}")

    return history, best_score

# =============================
# 8. PPO Baseline
# =============================
def run_ppo():
    env = make_env()
    model = PPO("MlpPolicy", env, seed=SEED, verbose=0)
    history = []

    for i in range(10):  # Reduced from 30 to 10
        model.learn(1000)  # Reduced from 2000 to 1000
        mean_reward, _ = evaluate_policy(model, env, 5)  # Reduced from 30 to 5
        history.append(mean_reward)
        print(f"PPO Step {i}: {mean_reward:.2f}")

    env.close()
    return history, history[-1]

# =============================
# 9. Single Run (instead of 5 runs)
# =============================
print("===== SINGLE RUN =====")
env = make_env()
policy = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0])
dim = len(flatten_params(policy))
env.close()

print("Running DE...")
de_history, de_final = run_de(policy, dim)

print("Running PSO...")
pso_history, pso_final = run_pso(policy, dim)

print("Running SA...")
sa_history, sa_final = run_sa(policy, dim)

print("Running PPO...")
ppo_history, ppo_final = run_ppo()

# =============================
# 10. Plot Results
# =============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(de_history, label='DE', marker='o')
plt.plot(pso_history, label='PSO', marker='s')
plt.plot(sa_history, label='SA', marker='^')
plt.plot(ppo_history, label='PPO', marker='d')
plt.xlabel('Generation/Step')
plt.ylabel('Fitness')
plt.title('Convergence Curves')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
labels = ["DE", "PSO", "SA", "PPO"]
finals = [de_final, pso_final, sa_final, ppo_final]
plt.bar(labels, finals)
plt.ylabel("Final Fitness")
plt.title("Final Performance")
plt.grid(True)

plt.tight_layout()
plt.savefig('results_fast.png')
plt.show()

print(f"\nFinal Results:")
print(f"DE: {de_final:.2f}")
print(f"PSO: {pso_final:.2f}")
print(f"SA: {sa_final:.2f}")
print(f"PPO: {ppo_final:.2f}")