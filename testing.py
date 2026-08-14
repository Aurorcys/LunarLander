import gymnasium as gym
import numpy as np
import pickle

class NumPyActor:
    def __init__(self, state_dim, action_dim):
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.W3, self.b3 = None, None
        self.log_std = None

    def load_weights(self, weights):
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]
        self.W3 = weights["W3"]
        self.b3 = weights["b3"]
        self.log_std = weights["log_std"]

    def forward(self, state, deterministic=True):
        if state.ndim == 1:
            state = state.reshape(1, -1)
    
        z1 = state @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)
        mean = a2 @ self.W3 + self.b3
        
        if deterministic:
            return mean.flatten()
        
        std = np.exp(np.clip(self.log_std, -2.0, 0.0))
        return np.random.normal(mean, std).flatten()



def stress_test_all_ranks(file_path, num_episodes):
    with open(file_path, "rb") as f:
        top5data = pickle.load(f)

    #descending
    top5data.sort(key=lambda x: x[0], reverse=True)

    env = gym.make("LunarLanderContinuous-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = NumPyActor(state_dim, action_dim)

    results = []

    print(f"Running {num_episodes} episodes for all 5 ranks")

    for rank, (train_ret, ep_num, weights) in enumerate(top5data, 1):
        actor.load_weights(weights)

        returns = []
        landings_solved = 0  
        crashes = 0          
        episode_lengths = []

        print(f"Testing Rank {rank} (Train Ep {ep_num} | Train Ret: {train_ret:.2f})")

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = actor.forward(state, deterministic=True)
                clamped_action = np.clip(action, -1.0, 1.0)

                state, reward, terminated, truncated, _ = env.step(clamped_action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            returns.append(total_reward)
            episode_lengths.append(steps)

            if total_reward >= 200:
                landings_solved += 1
            elif total_reward < 0:
                crashes += 1

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        max_ret = np.max(returns)
        min_ret = np.min(returns)
        solve_rate = (landings_solved / num_episodes) * 100
        crash_rate = (crashes / num_episodes) * 100
        avg_steps = np.mean(episode_lengths)

        results.append({
            "rank": rank,
            "train_ep": ep_num,
            "train_ret": train_ret,
            "test_mean": mean_ret,
            "test_std": std_ret,
            "max_ret": max_ret,
            "min_ret": min_ret,
            "solve_rate": solve_rate,
            "crash_rate": crash_rate,
            "avg_steps": avg_steps
        })

    env.close()

    print(f"{'Rank':<5} | {'Ep':<5} | {'Mean Test Ret':<16} | {'Max Ret':<9} | {'Min Ret':<9} | {'Solve %':<9} | {'Crash %':<9}")
    for r in results:
        mean_str = f"{r['test_mean']:.1f} ± {r['test_std']:.1f}"
        print(f"{r['rank']:<5} | {r['train_ep']:<5} | {mean_str:<16} | {r['max_ret']:<9.1f} | {r['min_ret']:<9.1f} | {r['solve_rate']:<8.1f}% | {r['crash_rate']:<8.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    pkl_path = "top5_actor_weights (2).pkl"
    stress_test_all_ranks(pkl_path, num_episodes=100)