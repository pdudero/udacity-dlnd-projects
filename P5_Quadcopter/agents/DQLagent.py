import numpy as np
import tensorflow as tf
from agents.DQLnetwork import QNetwork
from agents.replaybuffer import ReplayBuffer

class DQLagent():
    """Reinforcement Learning agent that learns using a DQL network."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size

        # Exploration parameters
        self.decay_max   = 1.0              # exploration probability at start
        self.decay_min   = 0.01             # minimum exploration probability 
        self.decay_rate  = 0.0001           # exponential decay rate for exploration prob
        self.decay_step  = np.exp(-self.decay_rate)
        self.decay_range = self.decay_max - self.decay_min
        self.decay_factor = 1.
        self.explore_p = self.decay_max

        # Network parameters
        self.learning_rate = 0.0001         # Q-network learning rate
        #self.learning_rate = 0.001         # Q-network learning rate

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor

        # Score tracker and learning parameters
        self.best_score = -np.inf
        self.score = -np.inf
        self.loss = 0

        self.qnet = QNetwork(self.state_size, self.action_size, name='main', learning_rate=self.learning_rate)

        # Episode variables
        self.reset_episode()
        
    def reset_episode(self,new_tgt_pos=None):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset(new_tgt_pos)
        self.last_state = state
        return state

    def act(self, tfsess, state):
        """Returns actions for given state(s) as per current policy."""
        # Explore or Exploit
        if len(self.memory) > self.batch_size:
            # epsilon-greedy policy:
            self.decay_factor *= self.decay_step
            self.explore_p = self.decay_min + (self.decay_range*self.decay_factor) 
            if self.explore_p > np.random.rand():
                # Make a random action
                actions = np.random.randint(0,self.action_size)
            else:
                # Get actions from Q-network
                feed = {self.qnet.inputs_: state.reshape((1, *state.shape))}
                Qs = tfsess.run(self.qnet.output, feed_dict=feed)
                actions = np.argmax(Qs)
        else:
            # pick actions equi-probablistically
            actions = np.random.randint(0,self.action_size)
        return actions

    def step(self,
             tfsess,
             action,     # int
             reward,     # np.ndarray (action_repeat,)
             next_state, # np.ndarray (state_size*action_repeat,)
             done):      # bool
         # Save experience / reward
        self.memory.add(self.last_state, action, np.mean(reward), next_state, done)

        # Save experience / reward
        self.total_reward += np.mean(reward)
        self.count += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(tfsess,experiences)

        # Roll over last state and action
        self.last_state = next_state

    def learn(self, tfsess, expbatch):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states      = np.array([e.state for e in expbatch if e is not None])
        actions     = np.array([e.action for e in expbatch if e is not None]).astype(np.float32)
        rewards     = np.array([e.reward for e in expbatch if e is not None]).astype(np.float32)
        dones       = np.array([e.done for e in expbatch if e is not None]).astype(np.uint8)
        next_states = np.array([e.next_state for e in expbatch if e is not None])

        # Train network
        target_Qs = tfsess.run(self.qnet.output, feed_dict={self.qnet.inputs_: next_states})
            
        # Set target_Qs to 0 for states where episode ends
        target_Qs[dones] = np.zeros(self.action_size)
        
        targets = rewards + self.gamma * np.max(target_Qs, axis=1)

        self.loss, _ = tfsess.run([self.qnet.loss, self.qnet.opt],
                                   feed_dict={self.qnet.inputs_: states,
                                              self.qnet.targetQs_: targets,
                                              self.qnet.actions_: actions})

        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
