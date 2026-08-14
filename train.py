import gymnasium as gym
import numpy as np
import pickle


class NumPyAdam:
    def __init__(self, params_dict, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, max_grad_norm=0.5):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.t = 0
        
        self.m = {k: np.zeros_like(v) for k, v in params_dict.items()}
        self.v = {k: np.zeros_like(v) for k, v in params_dict.items()}

    def step(self, params_dict, grads_dict):
        self.t += 1
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads_dict.values()))
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-6)
            grads_dict = {k: v * scale for k, v in grads_dict.items()}

        lr_t = self.lr * (np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t))

        for k in params_dict.keys():
            g = grads_dict[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)
            
            params_dict[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)



class NumPyActor:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.W1 = np.random.randn(state_dim, 64) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 64))
        self.W3 = np.random.randn(64, action_dim) * 0.01
        self.b3 = np.zeros((1, action_dim))
        
        self.log_std = np.full((1, action_dim), -0.7)

        self.params = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "log_std": self.log_std
        }
        self.optimizer = NumPyAdam(self.params, lr=lr)

    def forward(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
    
        self.z1 = state @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        self.mean = self.a2 @ self.W3 + self.b3
        
        
        self.std = np.exp(np.clip(self.log_std, -2.5, -0.5))
        return self.mean, self.std

    def backward(self, d_mean, d_log_std, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)

        d_W3 = self.a2.T @ d_mean
        d_b3 = np.sum(d_mean, axis=0, keepdims=True)
        
        d_a2 = d_mean @ self.W3.T
        d_z2 = d_a2 * (1 - self.a2 ** 2)
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (1 - self.a1 ** 2)
        d_W1 = state.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = {
            "W1": d_W1, "b1": d_b1,
            "W2": d_W2, "b2": d_b2,
            "W3": d_W3, "b3": d_b3,
            "log_std": d_log_std
        }

        self.optimizer.step(self.params, grads)


class NumPyCritic:
    def __init__(self, state_dim, lr=1e-3):
        self.W1 = np.random.randn(state_dim, 64) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 64))
        self.W3 = np.random.randn(64, 1) * 0.01
        self.b3 = np.zeros((1, 1))

        self.params = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3
        }
        self.optimizer = NumPyAdam(self.params, lr=lr)

    def forward(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        self.z1 = state @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        self.value = self.a2 @ self.W3 + self.b3
        return self.value

    def backward(self, d_val, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        d_W3 = self.a2.T @ d_val
        d_b3 = np.sum(d_val, axis=0, keepdims=True)
        
        d_a2 = d_val @ self.W3.T
        d_z2 = d_a2 * (1 - self.a2 ** 2)
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (1 - self.a1 ** 2)
        d_W1 = state.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = {
            "W1": d_W1, "b1": d_b1,
            "W2": d_W2, "b2": d_b2,
            "W3": d_W3, "b3": d_b3
        }

        self.optimizer.step(self.params, grads)



def compute_log_prob(action, mean, std):
    var = std ** 2
    log_prob = -0.5 * ((action - mean) ** 2) / (var + 1e-8) - np.log(std + 1e-8) - 0.5 * np.log(2 * np.pi)
    return np.sum(log_prob, axis=-1, keepdims=True)



#train
def train():
    env = gym.make("LunarLanderContinuous-v3")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = NumPyActor(state_dim, action_dim, lr=3e-4)
    critic = NumPyCritic(state_dim, lr=1e-3)

    episodes = 500
    steps_per_batch = 2048
    gamma = 0.99
    gae_lambda = 0.95
    ppo_clip = 0.2
    
    top5weights = []

    print("Training\n")
    print(f"{'Episode':<10} | {'Avg Return':<12} | {'Std (Thrust)':<15} | {'Std (Gimbal)':<15}")

    for episode in range(episodes + 1):
        entropy_coeff = 0.005 * (1.0 - (episode / episodes))
        
        states, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []
        state, _ = env.reset()
        episode_returns = []
        curr_return = 0

        
        for _ in range(steps_per_batch):
            mean, std = actor.forward(state)
            action = np.random.normal(mean, std).flatten()
            #clamp from 1 to 0.8
            clamped_action = np.clip(action, -0.8, 0.8)
            
            log_prob = compute_log_prob(action, mean, std).item()
            val = critic.forward(state).item()

            next_state, reward, terminated, truncated, _ = env.step(clamped_action)
            done = terminated or truncated

            states.append(state.copy())
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            values.append(val)
            dones.append(done)

            curr_return += reward
            
            if done:
                episode_returns.append(curr_return)
                curr_return = 0
                state, _ = env.reset()
            else:
                state = next_state

        #gae
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs).reshape(-1, 1)
        
        last_val = critic.forward(state).item()
        values_extended = np.append(values, last_val)
        
        advantages = np.zeros(len(rewards))
        last_gae = 0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values_extended[t + 1] * non_terminal - values_extended[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            
        returns = advantages + np.array(values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = advantages.reshape(-1, 1)

        #ppo update
        batch_size = 64
        indices = np.arange(len(states))
        
        for _ in range(10): 
            np.random.shuffle(indices)
            for start in range(0, len(states), batch_size):
                mb_idx = indices[start:start + batch_size]
        
                s = states[mb_idx]
                a = actions[mb_idx]
                old_lp = old_log_probs[mb_idx]
                adv = advantages[mb_idx]
                ret = returns[mb_idx].reshape(-1, 1)
        
                mean, std = actor.forward(s)
                curr_lp = compute_log_prob(a, mean, std)
        
                ratio = np.exp(curr_lp - old_lp)
                
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
                
                
                unclipped_mask = (surr1 <= surr2) | ((ratio >= 1.0 - ppo_clip) & (ratio <= 1.0 + ppo_clip))
                
             
                eff_adv = np.where(unclipped_mask, -adv * ratio, 0.0)
        
               
                d_mean = (eff_adv * (a - mean) / (std ** 2 + 1e-8)) / len(mb_idx)
                
                d_log_std_per_sample = eff_adv * (((a - mean) ** 2) / (std ** 2 + 1e-8) - 1.0)
                
               
                d_log_std = np.mean(d_log_std_per_sample, axis=0, keepdims=True) - entropy_coeff
        
                actor.backward(d_mean, d_log_std, s)
        
            
                val = critic.forward(s)
                d_val = (val - ret) / len(mb_idx)
                critic.backward(d_val, s)

        if episode % 10 == 0:
            avg_ret = np.mean(episode_returns) if len(episode_returns) > 0 else 0.0
            print(f"{episode:<10} | {avg_ret:<12.2f} | {std[0, 0]:<15.4f} | {std[0, 1]:<15.4f}")
            
            if len(top5weights) < 5 or avg_ret > top5weights[0][0]:
                weights_snapshot = {
                    "W1": actor.W1.copy(), "b1": actor.b1.copy(),
                    "W2": actor.W2.copy(), "b2": actor.b2.copy(),
                    "W3": actor.W3.copy(), "b3": actor.b3.copy(),
                    "log_std": actor.log_std.copy()
                }
                top5weights.append((avg_ret, episode, weights_snapshot))
                top5weights.sort(key=lambda x: x[0])
                if len(top5weights) > 5:
                    top5weights.pop(0)

    with open("top5_actor_weights.pkl", "wb") as f:
        pickle.dump(top5weights, f)
    print("\nSaved Top 5 models to 'top5_actor_weights.pkl'")

    return actor


if __name__ == "__main__":
    trained_actor = train()