from Simulation import run_simulation, read_simulation_data
from Signalverarbeitung import process_data, save_data
from Plot import plot_time_domain

# Define the range and step for Ton
ton_start = 480e-9
ton_end = 580e-9
ton_step = 30e-9

# Zusätzliche Parameter
tperiod = 1e-6  # Periodendauer
trise = 5e-9    # Anstiegszeit
tfall = 5e-9    # Abfallzeit

# Run a simulation and read data
path_raw = run_simulation(ton_start, ton_end, ton_step, tperiod, trise, tfall)

# Simulation durchführen
path_raw = run_simulation(ton_start, ton_end, ton_step, tperiod, trise, tfall)

if path_raw:
    # Read data
    t_LT, uds_LT = read_simulation_data(path_raw)

    # Process data
    t, uds = process_data(t_LT, uds_LT)

    # Save data
    save_data(t, uds)

    # Plot data
    plot_time_domain(t, uds, 1e-6, 1e-6 / 50001)
else:
    print("Simulation failed")

