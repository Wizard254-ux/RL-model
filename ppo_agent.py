from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create environment
env = make_vec_env("CartPole-v1", n_envs=1)

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_cartpole")

# Load model
model = PPO.load("ppo_cartpole")

# Test
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
