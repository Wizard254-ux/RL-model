from flask import Flask, request, jsonify
from stable_baselines3 import PPO
import gym

app = Flask(__name__)

# Load environment and model
env = gym.make("CartPole-v1")
model = PPO.load("ppo_cartpole")

@app.route("/predict", methods=["POST"])
def predict():
    obs = request.json.get("observation")
    if obs is None:
        return jsonify({"error": "Missing observation"}), 400

    action, _ = model.predict(obs)
    return jsonify({"action": int(action)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
