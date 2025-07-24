import os
import json
import logging
import threading
import time
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from collections import deque
import redis
import pickle
import gym

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced Metrics
EPISODE_COUNTER = Counter('rl_episodes_total', 'Total episodes completed', ['agent_id'])
REWARD_HISTOGRAM = Histogram('rl_episode_rewards', 'Episode rewards', ['agent_id'])
INFERENCE_HISTOGRAM = Histogram('rl_inference_time_seconds', 'Inference time')
TRAINING_LOSS = Gauge('rl_training_loss', 'Current training loss', ['agent_id'])
GPU_UTILIZATION = Gauge('rl_gpu_utilization_percent', 'GPU utilization', ['agent_id'])
EXPERIENCE_BUFFER_SIZE = Gauge('rl_experience_buffer_size', 'Experience buffer size')



class RedisExperienceBuffer:
    """Redis-based experience buffer for distributed mode"""
    
    def __init__(self, redis_client, max_size=100000):
        self.redis_client = redis_client
        self.max_size = max_size
        self.local_buffer = deque(maxlen=1000)
        
    def add_experience(self, state, action, reward, next_state, done, agent_id):
        experience = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': int(action),
            'reward': float(reward),
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'done': bool(done),
            'agent_id': agent_id,
            'timestamp': time.time()
        }
        self.local_buffer.append(experience)
        
        if len(self.local_buffer) >= 10:
            self._flush_to_redis()
    
    def _flush_to_redis(self):
        try:
            pipe = self.redis_client.pipeline()
            for exp in self.local_buffer:
                pipe.lpush('experience_buffer', pickle.dumps(exp))
            pipe.execute()
            self.local_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush to Redis: {e}")

class RedisParameterServer:
    """Redis-based parameter server"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.version = 0
        
    def update_parameters(self, model, agent_id):
        try:
            state_dict = model.policy.state_dict()
            serialized_params = {}
            
            for key, tensor in state_dict.items():
                serialized_params[key] = tensor.cpu().numpy().tobytes()
            
            self.version += 1
            param_data = {
                'parameters': serialized_params,
                'version': self.version,
                'agent_id': agent_id,
                'timestamp': time.time()
            }
            
            self.redis_client.set('global_parameters', pickle.dumps(param_data))
            logger.info(f"Updated Redis parameters v{self.version}")
            return self.version
        except Exception as e:
            logger.error(f"Failed to update Redis parameters: {e}")
            return None
    
    def get_latest_parameters(self):
        try:
            param_data = self.redis_client.get('global_parameters')
            if not param_data:
                return None, 0
            
            data = pickle.loads(param_data)
            state_dict = {}
            for key, tensor_bytes in data['parameters'].items():
                tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32)
                state_dict[key] = torch.from_numpy(tensor_array)
            
            return state_dict, data['version']
        except Exception as e:
            logger.error(f"Failed to get Redis parameters: {e}")
            return None, 0


class LocalExperienceBuffer:
    """Local experience buffer for standalone mode"""
    
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add_experience(self, state, action, reward, next_state, done, agent_id):
        """Add experience to local buffer"""
        experience = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': int(action),
            'reward': float(reward),
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'done': bool(done),
            'agent_id': agent_id,
            'timestamp': time.time()
        }
        self.buffer.append(experience)
        EXPERIENCE_BUFFER_SIZE.set(len(self.buffer))
    
    def sample_batch(self, batch_size=32):
        """Sample batch of experiences from local buffer"""
        if len(self.buffer) < batch_size:
            return None
        
        # Sample random experiences
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        # Convert to numpy arrays
        states = np.array([exp['state'] for exp in experiences])
        actions = np.array([exp['action'] for exp in experiences])
        rewards = np.array([exp['reward'] for exp in experiences])
        next_states = np.array([exp['next_state'] for exp in experiences])
        dones = np.array([exp['done'] for exp in experiences])
        
        return states, actions, rewards, next_states, dones

class LocalParameterServer:
    """Local parameter server for standalone mode"""
    
    def __init__(self):
        self.version = 0
        self.parameters = None
        
    def update_parameters(self, model, agent_id):
        """Store model parameters locally"""
        try:
            self.parameters = model.policy.state_dict()
            self.version += 1
            logger.info(f"Updated local parameters v{self.version} from agent {agent_id}")
            return self.version
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return None
    
    def get_latest_parameters(self):
        """Get latest local parameters"""
        return self.parameters, self.version

class EfficientDistributedAgent:
    """Highly optimized RL agent - standalone mode"""
    
    def __init__(self):
        self.agent_id = os.getenv('HOSTNAME', f'agent_{np.random.randint(1000, 9999)}')
        self.app = Flask(__name__)
        
        # Initialize components
        self.setup_connections()
        self.setup_environment()
        self.setup_model()
        self.setup_distributed_training()
        self.setup_routes()
        
        # Training state
        self.training_active = False
        self.current_episode = 0
        self.best_reward = float('-inf')
        
        # Performance monitoring
        self.performance_window = deque(maxlen=100)
        
    def setup_connections(self):
            """Initialize Redis connection"""
            try:
                # Redis Cloud connection
                self.redis_conn = redis.Redis(
                    host='redis-16946.c13.us-east-1-3.ec2.redns.redis-cloud.com',
                    port=16946,
                    decode_responses=False,  # Keep False for binary data
                    username="default",
                    password="7WJV1oTJBWRqOR3eGiA6rGrdj93zCcB0"
                )
                
                # Test connection
                self.redis_conn.ping()
                logger.info("Connected to Redis Cloud successfully")
                
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.redis_conn = None
            
            self.db_conn = None  # No PostgreSQL
    
    def setup_environment(self):
            """Initialize optimized environment"""
            # Fix numpy random seed issue
            np.random.seed(42)
            
            env_name = os.getenv('ENV_NAME', 'CartPole-v1')
            n_envs = int(os.getenv('N_ENVS', 4))  # Reduce for testing
            
            # Create vectorized environment with explicit seed
            self.env = make_vec_env(env_name, n_envs=n_envs, seed=42)
            self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
            
            # Get environment info
            self.obs_space = self.env.observation_space
            self.action_space = self.env.action_space
            
            logger.info(f"Created {n_envs} normalized environments of {env_name}")
    def setup_model(self):
        """Initialize optimized model with better hyperparameters"""
        model_type = os.getenv('MODEL_TYPE', 'PPO')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Optimized hyperparameters
        if model_type == 'PPO':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                device=device,
                tensorboard_log=f"./logs/{self.agent_id}/",
                policy_kwargs=dict(
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                    activation_fn=nn.ReLU
                )
            )
        elif model_type == 'DQN':
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                verbose=1,
                device=device,
                tensorboard_log=f"./logs/{self.agent_id}/"
            )
        
        logger.info(f"Initialized {model_type} model on {device}")
    
    def setup_distributed_training(self):
                """Initialize training components"""
                # Set synchronization settings FIRST
                self.sync_frequency = int(os.getenv('SYNC_FREQUENCY', 100))
                self.current_version = 0
                
                if self.redis_conn:
                    # Use Redis for distributed features
                    self.experience_buffer = RedisExperienceBuffer(self.redis_conn)
                    self.parameter_server = RedisParameterServer(self.redis_conn)
                    logger.info("Initialized Redis-based distributed training")
                else:
                    # Fallback to local
                    self.experience_buffer = LocalExperienceBuffer()
                    self.parameter_server = LocalParameterServer()
                    logger.info("Fallback to local training components")
                
        
    def setup_routes(self):
        """Define optimized API routes"""
        
        @self.app.route("/health", methods=["GET"])
        def health():
            gpu_available = torch.cuda.is_available()
            gpu_memory = None
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            return jsonify({
                "status": "running",
                "agent_id": self.agent_id,
                "device": str(self.model.device),
                "mode": "standalone",
                "training_active": self.training_active,
                "current_episode": self.current_episode,
                "best_reward": self.best_reward,
                "gpu_available": gpu_available,
                "gpu_memory_mb": gpu_memory // (1024*1024) if gpu_memory else None
            })
        
        @self.app.route("/predict", methods=["POST"])
        def predict():
            start_time = time.time()
            try:
                data = request.json
                observation = np.array(data["observation"])
                deterministic = data.get("deterministic", True)
                
                # Reshape if needed
                if len(observation.shape) == 1:
                    observation = observation.reshape(1, -1)
                
                action, _ = self.model.predict(observation, deterministic=deterministic)
                
                # Record inference time
                inference_time = time.time() - start_time
                INFERENCE_HISTOGRAM.observe(inference_time)
                
                return jsonify({
                    "action": int(action[0]) if hasattr(action, '__len__') else int(action),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "agent_id": self.agent_id
                })
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({"error": str(e)}), 400
        
        @self.app.route("/train", methods=["POST"])
        def train():
            if self.training_active:
                return jsonify({"error": "Training already in progress"}), 400
            
            try:
                data = request.json
                timesteps = data.get("timesteps", 50000)
                save_model = data.get("save_model", True)
                
                # Start training in background thread
                training_thread = threading.Thread(
                    target=self._train_async,
                    args=(timesteps, save_model)
                )
                training_thread.start()
                
                return jsonify({
                    "status": "training_started",
                    "timesteps": timesteps,
                    "agent_id": self.agent_id
                })
                
            except Exception as e:
                logger.error(f"Training start error: {e}")
                return jsonify({"error": str(e)}), 400
        
        @self.app.route("/stop_training", methods=["POST"])
        def stop_training():
            self.training_active = False
            return jsonify({"status": "training_stopped", "agent_id": self.agent_id})
        
        @self.app.route("/sync_model", methods=["POST"])
        def sync_model():
            try:
                success = self._synchronize_model()
                return jsonify({
                    "status": "synchronized" if success else "failed",
                    "agent_id": self.agent_id,
                    "current_version": self.current_version
                })
            except Exception as e:
                logger.error(f"Sync error: {e}")
                return jsonify({"error": str(e)}), 400
        
        @self.app.route("/metrics", methods=["GET"])
        def metrics():
            return generate_latest()
        
        @self.app.route("/performance", methods=["GET"])
        def performance():
            if not self.performance_window:
                return jsonify({"error": "No performance data available"}), 404
            
            recent_rewards = list(self.performance_window)
            return jsonify({
                "agent_id": self.agent_id,
                "episodes_completed": len(recent_rewards),
                "mean_reward": np.mean(recent_rewards),
                "std_reward": np.std(recent_rewards),
                "best_reward": max(recent_rewards),
                "worst_reward": min(recent_rewards),
                "recent_rewards": recent_rewards[-10:]
            })
    
    def _train_async(self, timesteps, save_model):
        """Asynchronous training - standalone mode"""
        self.training_active = True
        logger.info(f"Starting training for {timesteps} timesteps")
        
        try:
            # Custom callback for training
            callback = StandaloneTrainingCallback(
                self.experience_buffer,
                self.parameter_server,
                self.agent_id,
                self.sync_frequency,
                self.performance_window
            )
            
            # Train the model
            self.model.learn(
                total_timesteps=timesteps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Save model if requested
            if save_model:
                os.makedirs("/app/models", exist_ok=True)
                model_path = f"/app/models/{self.agent_id}_model"
                self.model.save(model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Update local parameters
            self.parameter_server.update_parameters(self.model, self.agent_id)
            
        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            self.training_active = False
            logger.info("Training completed")
    
    def _synchronize_model(self):
        """Synchronize model - standalone mode (local only)"""
        try:
            params, version = self.parameter_server.get_latest_parameters()
            
            if params is None or version <= self.current_version:
                logger.debug("No new parameters to sync")
                return True
            
            # Load parameters into model
            self.model.policy.load_state_dict(params, strict=False)
            self.current_version = version
            
            logger.info(f"Synchronized to version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            return False
    
    def run(self):
        """Start the Flask application"""
        # Start Flask app
        port = int(os.getenv('PORT', 5000))
        self.app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

class StandaloneTrainingCallback(BaseCallback):
    """Enhanced callback for standalone training"""
    
    def __init__(self, experience_buffer, parameter_server, agent_id, sync_frequency, performance_window):
        super().__init__()
        self.experience_buffer = experience_buffer
        self.parameter_server = parameter_server
        self.agent_id = agent_id
        self.sync_frequency = sync_frequency
        self.performance_window = performance_window
        self.episode_count = 0
        self.last_sync = 0
    
    def _on_step(self) -> bool:
        # Sync parameters periodically (local only)
        if self.num_timesteps - self.last_sync >= self.sync_frequency:
            self.parameter_server.update_parameters(self.model, self.agent_id)
            self.last_sync = self.num_timesteps
        
        # Process episode information
        if len(self.model.ep_info_buffer) > 0:
            for episode_info in self.model.ep_info_buffer:
                reward = episode_info.get('r', 0)
                length = episode_info.get('l', 0)
                
                # Update metrics
                EPISODE_COUNTER.labels(agent_id=self.agent_id).inc()
                REWARD_HISTOGRAM.labels(agent_id=self.agent_id).observe(reward)
                
                # Store performance
                self.performance_window.append(reward)
                
                # Log training progress
                if self.episode_count % 10 == 0:
                    logger.info(f"Agent {self.agent_id} - Episode {self.episode_count}: "
                              f"Reward={reward:.2f}, Length={length}")
                
                self.episode_count += 1
            
            # Clear buffer to avoid reprocessing
            self.model.ep_info_buffer.clear()
        
        return True
    
    def _on_training_end(self) -> None:
        # Final parameter update
        self.parameter_server.update_parameters(self.model, self.agent_id)
        logger.info(f"Training completed for agent {self.agent_id}")

if __name__ == "__main__":
    agent = EfficientDistributedAgent()
    agent.run()