import os
import pickle
import base64
import imageio
import numpy as np
import gymnasium as gym
from IPython.display import HTML, display


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
        
        std = np.exp(np.clip(self.log_std, -2.5, -0.5))
        return np.random.normal(mean, std).flatten()



def record_all_ranks_and_embed(
    pkl_path="/kaggle/input/datasets/aurorcys/top5thirditeration/top5_actor_weights (2).pkl",
    fps=50
):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find pickle file at: {pkl_path}")

    print(f"Loading weights from Kaggle dataset: {pkl_path}\n")
    with open(pkl_path, "rb") as f:
        top5weights = pickle.load(f)

    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    video_cards_html = ""

    #loop 
    for rank_idx in range(5):
        output_mp4 = f"lunar_landing_rank{rank_idx + 1}_h264.mp4"

        
        if isinstance(top5weights, list) and rank_idx < len(top5weights):
            avg_ret, train_ep, weights = top5weights[rank_idx]
            rank_name = f"Rank {rank_idx + 1}"
        elif isinstance(top5weights, dict) and "weights" in top5weights:
            weights = top5weights["weights"]
            train_ep = top5weights.get("train_episode", f"Rank {rank_idx + 1}")
            avg_ret = top5weights.get("avg_return", "N/A")
            rank_name = f"Rank {rank_idx + 1}"
        else:
            weights = top5weights
            train_ep, avg_ret = "N/A", "N/A"
            rank_name = f"Rank {rank_idx + 1}"

        actor = NumPyActor(state_dim, action_dim)
        actor.load_weights(weights)

        print(f"Recording landing sequence for {rank_name} (Train Ep {train_ep} | Expected Ret: {avg_ret:.2f})...")

        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        frames = []

        frames.append(env.render())

        while not done:
            action = actor.forward(state, deterministic=True)
            clamped_action = np.clip(action, -1.0, 1.0)

            state, reward, terminated, truncated, _ = env.step(clamped_action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            frames.append(env.render())

        print(f"  Completed in {steps} steps | Score: {total_reward:.2f}")

        #html
        writer = imageio.get_writer(output_mp4, fps=fps, codec='libx264', pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        
        with open(output_mp4, "rb") as f:
            video_bytes = f.read()
            b64_video = base64.b64encode(video_bytes).decode("utf-8")

        
        video_cards_html += f"""
        <div style="flex: 1 1 45%; min-width: 320px; max-width: 500px; margin: 15px; padding: 15px; 
                    border: 1px solid #444; border-radius: 10px; background-color: #1e1e1e; color: #fff; text-align: center;">
            <h4 style="margin-top: 0; color: #4CAF50;">{rank_name} (Train Ep {train_ep})</h4>
            <p style="font-size: 14px; color: #ddd; margin-bottom: 10px;">Test Score: <b>{total_reward:.2f}</b> | Steps: {steps}</p>
            <video width="100%" height="auto" controls autoplay loop style="border-radius: 6px; border: 1px solid #555;">
                <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                Your browser does not support HTML5 video.
            </video>
            <br><br>
            <a href="data:video/mp4;base64,{b64_video}" download="{output_mp4}" 
               style="background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; 
                      font-weight: bold; border-radius: 5px; display: inline-block; font-size: 13px;">
                Download {output_mp4}
            </a>
        </div>
        """

    env.close()

    full_html = f"""
    <div style="text-align: center; font-family: sans-serif;">
        <h2 style="color: #333;">🎥 Top 5 PPO Actor Landing Showcase</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: flex-start;">
            {video_cards_html}
        </div>
    </div>
    """
    display(HTML(full_html))


if __name__ == "__main__":
    record_all_ranks_and_embed()