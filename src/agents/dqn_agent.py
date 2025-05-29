"""
Deep Q-Learning Agent for Trading

This module implements a DQN agent that can be used for both backtesting and live trading.
"""

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path

class DQNAgent:
    """
    Deep Q-Network Agent for trading
    
    Features:
    - Experience replay
    - Target network
    - Double DQN
    - Compatible with both backtesting and live trading
    """
    
    def __init__(self, 
                 state_size,
                 action_size=3,  # SHORT, HOLD, LONG
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 batch_size=32,
                 memory_size=10000):
        """
        Initialize the DQN agent
        
        Parameters:
        -----------
        state_size : int
            Dimension of the state space
        action_size : int
            Dimension of the action space (default: 3 for SHORT, HOLD, LONG)
        gamma : float
            Discount factor
        epsilon : float
            Exploration rate
        epsilon_min : float
            Minimum exploration rate
        epsilon_decay : float
            Decay rate for exploration
        learning_rate : float
            Learning rate for the optimizer
        batch_size : int
            Batch size for training
        memory_size : int
            Size of the replay memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build a neural network model for deep Q learning"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the primary model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose an action based on the current state"""
        # Exploration
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def predict(self, state):
        """Predict action for a given state (for compatibility with backtesting)"""
        return self.act(state, training=False)
    
    def replay(self, batch_size=None):
        """Train the model using experience replay"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state.reshape(1, -1), verbose=0)
            
            if done:
                target[0][action] = reward
            else:
                # Double DQN: Select action using the primary network
                a = np.argmax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
                # But evaluate it using the target network
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[0][action] = reward + self.gamma * t[a]
            
            # Train the network
            self.model.fit(state.reshape(1, -1), target, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights from file"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights to file"""
        self.model.save_weights(name)
        
    def get_state_from_features(self, features):
        """
        Convert feature dictionary to state vector
        
        Parameters:
        -----------
        features : dict
            Dictionary of features from AlpacaInterface.get_state_features()
            
        Returns:
        --------
        np.array : State vector for the agent
        """
        # Extract relevant features and normalize them
        # This is an example - adapt based on your specific features
        state = np.array([
            features.get('returns', 0),
            features.get('ret_2', 0),
            features.get('ret_5', 0),
            features.get('ret_10', 0),
            features.get('position', 0) / 100,  # Normalize position
            features.get('unrealized_pl', 0) / 1000  # Normalize P&L
        ])
        
        return state
