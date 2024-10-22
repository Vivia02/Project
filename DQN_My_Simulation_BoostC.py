import numpy as np
from DQN import DQNAgent
from My_Simulation_BoostConverter import run_simulation, read_simulation_data, process_data
import matplotlib.pyplot as plt

class BoostConverterEnv:
    def __init__(self, state_size, action_size, target_dB=10.0):
        self.state_size = state_size
        self.action_size = action_size
        self.target_dB = target_dB
        self.state = None
        self.current_step = 0
        self.amplitude = 1.8  # Starte mit der Anfangsamplitude
        self.phase = -150  # Starte mit der Anfangsphase (-150°)
        self.steps_per_episode = 10  # Begrenzte Schritte pro Episode

    def reset(self):
        # Reset the environment and simulation
        print("Simulation reset... Amplitude: {:.1f}, Phase: {}".format(self.amplitude, self.phase))

        self.current_step = 0
        ton_start = 480e-9
        ton_end = 580e-9
        ton_step = 30e-9
        tperiod = 1e-6
        trise = 5e-9
        tfall = 5e-9

        # Run initial simulation
        path_raw = run_simulation(ton_start, ton_end, ton_step, tperiod, trise, tfall)
        if path_raw:
            t_LT, uds_LT = read_simulation_data(path_raw)
            t, uds = process_data(t_LT, uds_LT)
            self.state = [t, uds]
            return np.array(self.state).reshape(1, -1)
        else:
            print("Simulation failed in reset")
            return None

    def step(self, action):
        # Define actions to modify the parameters in the simulation
        ton_start = 480e-9
        ton_end = 580e-9
        ton_step = 30e-9
        tperiod = 1e-6
        trise = 5e-9
        tfall = 5e-9

        # Amplituden- und Phasenanpassungen, sinnvoll innerhalb der Grenzen
        if action == 0:
            ton_step += 30e-9  # Increase ton_step
        elif action == 1:
            ton_step -= 30e-9  # Decrease ton_step
        elif action == 2:  # Amplitude erhöhen (auf maximal 2.6 V)
            if self.amplitude < 2.6:
                self.amplitude += 0.2
        elif action == 3:  # Amplitude verringern (auf minimal 1.0 V)
            if self.amplitude > 1.0:
                self.amplitude -= 0.2
        elif action == 4:  # Phase wechseln zwischen -150° und 120°
            if self.phase == -150:
                self.phase = 120
            else:
                self.phase = -150

        # Run the simulation with modified parameters
        path_raw = run_simulation(ton_start, ton_end, ton_step, tperiod, trise, tfall)
        if path_raw:
            t_LT, uds_LT = read_simulation_data(path_raw)
            t, uds = process_data(t_LT, uds_LT)

            # Berechne den Reward basierend auf dem dB-Unterschied
            V_in = 1.0
            V_out = np.mean(uds)
            current_dB = 20 * np.log10(V_out / V_in)
            reward = -np.square(current_dB - self.target_dB)

            # Schwingungsstrafe (Stabilitätsstrafe) basierend auf der Varianz
            stability_penalty = np.var(uds)
            reward -= stability_penalty * 10  # Bestrafe größere Schwingungen

            # Update the state
            self.state = [t, uds]
            done = False
            self.current_step += 1

            if self.current_step >= self.steps_per_episode:  # Episode endet nach der festgelegten Anzahl von Schritten
                done = True

            # Debug-Ausgabe, um den Lernprozess zu verfolgen
            print(f"Step {self.current_step}: Action taken -> {action}, Amplitude: {self.amplitude}, Phase: {self.phase}")
            print(f"Reward calculated: {reward}")

            return np.array(self.state).reshape(1, -1), reward, done
        else:
            print("Simulation failed in step")
            return None, -1, True

    def render(self):
        t, uds = self.state
        plt.figure(figsize=(10, 6))
        plt.plot(t, uds, 'b', label=f'Output Signal Uds (Step {self.current_step})')
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Drain Signal Uds (Step {self.current_step})")
        plt.legend()
        plt.grid(True)
        plt.show()

# Jetzt kannst du diesen Agenten verwenden und trainieren

# DQN-Trainingsparameter
state_size = 2  # Zeit und Spannung
action_size = 5  # Anpassen von Ton, Amplitude, und Phase
env = BoostConverterEnv(state_size, action_size)

agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 50
batch_size = 32

for e in range(episodes):
    print(f"Starte Episode {e+1}/{episodes}...")

    state = env.reset()

    total_reward = 0
    for time in range(env.steps_per_episode):  # Begrenzte Schrittzahl pro Episode
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Visualisiere nach jeder Episode
    env.render()

    # Speichere das Modell nach jeder Episode
    agent.save(f"DQN_BoostConverter_episode_{e+1}.weights.h5")
