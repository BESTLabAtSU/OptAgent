import matplotlib.pyplot as plt
import numpy as np
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
import os
from pathlib import Path

# Set up matplotlib for publication-quality plots
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.figsize'] = (3, 1.5)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True

# Get project root path
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

# Load configuration
config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", "config_setup.json")
cm = ConfigurationManager(config_path)
env = BESTOptEnvironment(cm.config)

# Initialize data storage
timesteps = []
supply_air_flow_real_data = []
supply_air_flow_setpt_data = []

# Run Simulation
print("Starting simulation...")
for timestep in range(env.total_step):
    # Step the environment
    observations, done, info = env.step()

    # Get building and system IDs
    cluster_id = 'residential_cluster_1'
    building_id = 'SFH_1'
    hvac_system_id = env.building_system_map[building_id].get('thermal')

    # Get HVAC system module
    hvac_system = env.system_modules[hvac_system_id]

    # Get supply air flow parameters
    supply_air_flow_real = hvac_system.SA_flow_actual_m3s
    supply_air_flow_setpt = env.cluster_actions[cluster_id].thermal.system_actions[
        'hvac_system_1'].supply_airflow_setpoint_m3s

    # Store data
    timesteps.append(timestep)
    supply_air_flow_real_data.append(supply_air_flow_real)
    supply_air_flow_setpt_data.append(supply_air_flow_setpt)

    # Print progress every 100 steps
    if timestep % 100 == 0:
        print(f"Step {timestep}/{env.total_step}: "
              f"Flow Real={supply_air_flow_real:.4f} m³/s, "
              f"Flow Setpoint={supply_air_flow_setpt:.4f} m³/s")

    if done:
        break

print(f"\nSimulation completed: {len(timesteps)} steps")

# Convert to numpy arrays for easier manipulation
timesteps = np.array(timesteps)
supply_air_flow_real_data = np.array(supply_air_flow_real_data)
supply_air_flow_setpt_data = np.array(supply_air_flow_setpt_data)

# Create the publication-quality plot
fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=300)

# Plot the data
ax.plot(timesteps, supply_air_flow_setpt_data,
        label='Setpoint', color='#2E86AB', linestyle='--',
        linewidth=1.2, alpha=0.8)
ax.plot(timesteps, supply_air_flow_real_data,
        label='Actual', color='#A23B72', linestyle='-',
        linewidth=0.8, alpha=0.9)

# Customize the plot
ax.set_xlabel('Time Step', fontweight='normal', fontsize=7)
ax.set_ylabel('Flow Rate (m³/s)', fontweight='normal', fontsize=7)
# ax.set_title('Supply Air Flow Rate: Setpoint vs Actual', fontweight='normal', fontsize=7, pad=5)

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

# Customize legend - no frame
# legend = ax.legend(loc='upper right', frameon=False, fontsize=7)

# Set y-axis limits with some padding
y_min = min(min(supply_air_flow_real_data), min(supply_air_flow_setpt_data))
y_max = max(max(supply_air_flow_real_data), max(supply_air_flow_setpt_data))
y_range = y_max - y_min
ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

# Add minor ticks for better precision reading
ax.minorticks_on()
ax.tick_params(which='minor', length=3, width=0.5)
ax.tick_params(which='major', length=6, width=1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure in multiple formats for publication
print("\nSaving figures...")
fig.savefig('fan_flow_plot_paper.pdf', dpi=300, bbox_inches='tight', format='pdf')
fig.savefig('fan_flow_plot_paper.png', dpi=300, bbox_inches='tight', format='png')
fig.savefig('fan_flow_plot_paper.eps', dpi=300, bbox_inches='tight', format='eps')
print("Figures saved as: fan_flow_plot_paper.pdf, fan_constant.png, fan_flow_plot_paper.eps")

# Display the plot
plt.show()

# Print statistics
print("\n=== Supply Air Flow Statistics ===")
print(f"Actual Flow - Mean: {np.mean(supply_air_flow_real_data):.4f} m³/s")
print(f"Actual Flow - Std Dev: {np.std(supply_air_flow_real_data):.4f} m³/s")
print(f"Actual Flow - Min: {np.min(supply_air_flow_real_data):.4f} m³/s")
print(f"Actual Flow - Max: {np.max(supply_air_flow_real_data):.4f} m³/s")
print(f"\nSetpoint Flow - Mean: {np.mean(supply_air_flow_setpt_data):.4f} m³/s")
print(f"Setpoint Flow - Std Dev: {np.std(supply_air_flow_setpt_data):.4f} m³/s")
print(f"Setpoint Flow - Min: {np.min(supply_air_flow_setpt_data):.4f} m³/s")
print(f"Setpoint Flow - Max: {np.max(supply_air_flow_setpt_data):.4f} m³/s")

# Calculate tracking error
tracking_error = supply_air_flow_real_data - supply_air_flow_setpt_data
rmse = np.sqrt(np.mean(tracking_error ** 2))
mae = np.mean(np.abs(tracking_error))
print(f"\n=== Tracking Performance ===")
print(f"RMSE: {rmse:.6f} m³/s")
print(f"MAE: {mae:.6f} m³/s")
print(f"Max Absolute Error: {np.max(np.abs(tracking_error)):.6f} m³/s")