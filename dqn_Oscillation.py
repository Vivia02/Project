import gym
from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#stop
def trapez_puls(t, tr, tau, A):
    """
    Generates a trapezoidal pulse for the time vector `t` with
    a rise and fall time of `tr` and a pulse width of `tau`.
    The amplitude of the pulse is `A`.
    """
    Signal = np.zeros_like(t)  # Initialize the signal array
    T = t[-1] + (t[1] - t[0])
    tTrapez = t - T/2
    for mm in range(len(t)):
        if tTrapez[mm] > -(tau + tr) / 2 and tTrapez[mm] < -(tau - tr) / 2:
            Signal[mm] = A / tr * (tTrapez[mm] + (tau + tr) / 2)
        elif abs(tTrapez[mm]) <= (tau - tr) / 2:
            Signal[mm] = A
        elif tTrapez[mm] < (tau + tr) / 2 and tTrapez[mm] > (tau - tr) / 2:
            Signal[mm] = -A / tr * (tTrapez[mm] - (tau + tr) / 2)
    return Signal


class SignalEnv(Env):
    def __init__(self, fixed_frequency=1, episode_length=100):

        # Initialize environment parameters
        self.t = np.linspace(0, 10, 1000)
        self.tr = 2  # Rise and fall time
        self.tau = 5  # Pulse width
        self.A = 1  # Amplitude of the trapezoidal pulse
        self.f = fixed_frequency

        # Define the action space: increase, decrease, maintain (discrete actions)
        self.action_space = Discrete(3)  # 0: decrease, 1: maintain, 2: increase

        # Define the observation space (state space)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)

        # Store the episode length
        self.episode_length = episode_length
        self.current_step = 0

    def reset(self):
        # Reset the environment state for a new episode
        self.current_step = 0
        #self.phase_shift = 0.0
        self.trapezSignal = trapez_puls(self.t, self.tr, self.tau, self.A)
        self.sin_wave = 0.2 * np.sin(2 * np.pi * self.f * self.t)
        self.state = self.trapezSignal + self.sin_wave
        return self.state

    def step(self, action):

        # Apply the action and update the state
        self.current_step += 1

        if action == 0:  # Decrease phase
            self.sin_wave = 0.2 * np.sin(2 * np.pi * self.f * self.t - 1)
        elif action == 1:  # Maintain phase
            self.sin_wave = 0.2 * np.sin(2 * np.pi * self.f * self.t)
        elif action == 2:  # Increase phase
            self.sin_wave = 0.2 * np.sin(2 * np.pi * self.f * self.t + 1)

        # Update the state with the new trapezSignal
        self.trapezSignal = trapez_puls(self.t, self.tr, self.tau, self.A)
        self.state = self.trapezSignal + self.sin_wave

        # Calculate reward based on variance (less variance = more stable)
        reward = -np.var(self.state)

        # # Amplitude
        # target_amplitude = np.mean(self.trapezSignal)
        # current_amplitude = np.mean(self.state)
        # reward = -abs(current_amplitude - target_amplitude)

        # Determine if the episode is done
        done = self.current_step >= self.episode_length
        info = {}
        return self.state, reward, done, info


    def render(self, mode='human'):
        # Visualization of the current state
        plt.figure(figsize=(10, 6))
        # Plot trapezoidal signal
        plt.plot(self.t, self.trapezSignal, 'b--', label='Trapezoidal Pulse')
        # Plot combined signal
        plt.plot(self.t, self.state, 'k', label='Current Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Current Signal at f={self.f} Hz')
        plt.legend()
        plt.grid(True)

        # plt.draw()  # Draw the plot without blocking
        # plt.pause(0.001)  # Pause briefly to allow the plot to update
        # plt.clf()  # Clear the figure for the next episode's plot
        plt.show()

    def close(self):
        pass

class DQNAgent:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        self.model = self.build_model()
    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate= self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Select epsilon-greedy action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)         # take a random action (exploration)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])                # Best action (exploitation)

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        if not minibatch:
            return
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)  # Remove the extra dimension
        next_states = np.squeeze(next_states)  # Remove the extra dimension

        # Predict Q-values for next states
        q_values_next = self.model.predict_on_batch(next_states)
        max_q_values_next = np.amax(q_values_next)

        # Calculate targets: reward + gamma * max_q_value_next (if not done)
        targets = rewards + self.gamma * max_q_values_next * (1 - dones)

        # Predict Q-values for the current states
        targets_full = self.model.predict_on_batch(states)

        # Update the Q-values for the actions taken
        ind = np.arange(self.batch_size)
        targets_full[ind, actions] = targets

        # Train the model
        self.model.fit(states, targets_full, epochs=1, verbose=0)

        # # Apply epsilon decay after training
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)

# Initialize the environment and agent
env = SignalEnv(fixed_frequency=1, episode_length=100)
agent = DQNAgent(state_space=1000, action_space=3)

# Training loop
for episode in range(100):  # Number of episodes to train
    state = env.reset()
    state = np.reshape(state, [1, 1000])  # Reshape state for the neural network

    score = 0

    for step in range(env.episode_length):

        action = agent.act(state)  # Choose action based on current state
        next_state, reward, done, info = env.step(action)  # Apply action and get feedback
        score += reward
        next_state = np.reshape(next_state, [1, 1000])
        agent.remember(state, action, reward, next_state, done)  # Store experience
        state = next_state
        agent.replay()  # Train the model

        if done:
             break

             # Decay epsilon after each episode, not after each step
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Episode {episode + 1}, Total Reward: {score:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Render the environment's state at the end of each episode
    env.render()

    # save the model periodically (optional)
    if episode % 100 == 0:
        agent.save_model(f"dqn_model_{episode}.keras")

plt.close()







