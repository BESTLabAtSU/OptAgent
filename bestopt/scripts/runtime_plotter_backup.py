import matplotlib
matplotlib.use("QtAgg")  # Uncomment for interactive display
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from PIL import Image
import io

import matplotlib
matplotlib.use("QtAgg")  # For interactive display
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# ------------------------------------------------------------
# HVAC Single-Building Dashboard (clean + zero-safe + stable)
# ------------------------------------------------------------
import matplotlib
matplotlib.use("QtAgg")  # Change to "Agg" if running headless
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


def _to_float_or_nan(x):
    """Keep real zeros; convert only None to NaN."""
    return np.nan if x is None else float(x)


class HVACDashboard:
    """Dashboard for monitoring a single building's HVAC system."""

    def __init__(self, max_points=200, window_title="HVAC System Monitor"):
        # Global styling
        plt.rcParams.update({
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        })

        self.max_points = max_points

        # Data buffers
        self.timesteps = deque(maxlen=max_points)
        self.temperatures = deque(maxlen=max_points)
        self.hvac_thermal_loads = deque(maxlen=max_points)
        self.hvac_powers = deque(maxlen=max_points)
        self.cooling_setpoints = deque(maxlen=max_points)
        self.heating_setpoints = deque(maxlen=max_points)
        self.supply_air_temps_real = deque(maxlen=max_points)
        self.supply_air_temps_setpt = deque(maxlen=max_points)
        self.supply_air_flows_real = deque(maxlen=max_points)
        self.supply_air_flows_setpt = deque(maxlen=max_points)

        # Figure & axes
        plt.ion()
        self.fig = plt.figure(figsize=(7.8, 6.4), dpi=120)
        try:
            self.fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

        gs = self.fig.add_gridspec(3, 2, hspace=0.8, wspace=0.45)
        self.ax1 = self.fig.add_subplot(gs[0, :])   # Zone temp + setpoints
        self.ax2 = self.fig.add_subplot(gs[1, 0])   # Thermal load
        self.ax3 = self.fig.add_subplot(gs[1, 1])   # HVAC power
        self.ax4 = self.fig.add_subplot(gs[2, 0])   # SAT
        self.ax5 = self.fig.add_subplot(gs[2, 1])   # SAF

        # Lines
        self.temp_line, = self.ax1.plot([], [], color="#0072B2", lw=2, label="Zone Temperature")
        self.cool_setpt_line, = self.ax1.plot([], [], "--", color="gray", lw=1.5,
                                              label="Cooling Setpoint", drawstyle="steps-post")
        self.heat_setpt_line, = self.ax1.plot([], [], "--", color="gray", lw=1.5,
                                              label="Heating Setpoint", drawstyle="steps-post")

        self.thermal_load_line, = self.ax2.plot([], [], color="#E69F00", lw=2, label="Thermal Load")
        self.power_line, = self.ax3.plot([], [], color="#9900CC", lw=2, label="HVAC Power")

        self.sat_real_line, = self.ax4.plot([], [], "g-", lw=2, label="Actual SAT")
        self.sat_setpt_line, = self.ax4.plot([], [], "g--", lw=1.5, label="Setpoint SAT",
                                             alpha=0.85, drawstyle="steps-post")

        self.saf_real_line, = self.ax5.plot([], [], color="#56B4E9", lw=2, label="Actual Flow")
        self.saf_setpt_line, = self.ax5.plot([], [], "--", color="#56B4E9", lw=1.5,
                                             label="Setpoint Flow", alpha=0.85, drawstyle="steps-post")

        self._setup_axes()
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.08, hspace=0.85, wspace=0.45)
        plt.show(block=False)

    # --------- Appearance ---------
    def _setup_axes(self):
        for ax, title, ylabel in [
            (self.ax1, "Zone Temperature & Setpoints", "Temperature (°C)"),
            (self.ax2, "HVAC Thermal Load", "Thermal Load (W)"),
            (self.ax3, "HVAC Power Consumption", "Power (W)"),
            (self.ax4, "Supply Air Temperature", "Temperature (°C)"),
            (self.ax5, "Supply Air Flow Rate", "Flow Rate (m³/s)"),
        ]:
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="gray", alpha=0.25, linestyle="--")
            ax.set_facecolor("#f7f7f7")
            ax.legend(loc="upper right")

        self.ax2.set_xlabel("Timestep")
        self.ax3.set_xlabel("Timestep")
        self.ax4.set_xlabel("Timestep")
        self.ax5.set_xlabel("Timestep")

    # --------- Data ingest + redraw ---------
    def add_data_point(self, timestep, zone_temperature, hvac_thermal_load, hvac_power,
                       supervisory_cooling_setpoint=None, supervisory_heating_setpoint=None,
                       supply_air_temp_real=None, supply_air_temp_setpt=None,
                       supply_air_flow_real=None, supply_air_flow_setpt=None):
        """Append one point for all series and refresh plots."""
        self.timesteps.append(float(timestep))
        self.temperatures.append(float(zone_temperature))
        self.hvac_thermal_loads.append(float(hvac_thermal_load))
        self.hvac_powers.append(float(hvac_power))

        # Keep real zeros; only None -> NaN
        self.cooling_setpoints.append(_to_float_or_nan(supervisory_cooling_setpoint))
        self.heating_setpoints.append(_to_float_or_nan(supervisory_heating_setpoint))
        self.supply_air_temps_real.append(_to_float_or_nan(supply_air_temp_real))
        self.supply_air_temps_setpt.append(_to_float_or_nan(supply_air_temp_setpt))
        self.supply_air_flows_real.append(_to_float_or_nan(supply_air_flow_real))
        self.supply_air_flows_setpt.append(_to_float_or_nan(supply_air_flow_setpt))

        x = list(self.timesteps)
        self.temp_line.set_data(x, self.temperatures)
        self.cool_setpt_line.set_data(x, self.cooling_setpoints)
        self.heat_setpt_line.set_data(x, self.heating_setpoints)
        self.thermal_load_line.set_data(x, self.hvac_thermal_loads)
        self.power_line.set_data(x, self.hvac_powers)
        self.sat_real_line.set_data(x, self.supply_air_temps_real)
        self.sat_setpt_line.set_data(x, self.supply_air_temps_setpt)
        self.saf_real_line.set_data(x, self.supply_air_flows_real)
        self.saf_setpt_line.set_data(x, self.supply_air_flows_setpt)

        if len(x) > 2:
            self._autoscale_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0001)

    # --------- Stable autoscaling ---------
    def _autoscale_axes(self):
        x = list(self.timesteps)
        xmin, xmax = min(x), max(x)
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_xlim(xmin, xmax)

        # 1) Zone temperature: use temps + setpoints; keep a small margin
        t_all = np.array(list(self.temperatures), dtype=float)
        cool = np.array(list(self.cooling_setpoints), dtype=float)
        heat = np.array(list(self.heating_setpoints), dtype=float)
        tstack = np.concatenate([t_all, cool, heat])
        if not np.all(np.isnan(tstack)):
            tmin, tmax = np.nanmin(tstack), np.nanmax(tstack)
            self.ax1.set_ylim(tmin - 1.0, tmax + 1.0)

        # 2) Thermal load: symmetric cushion
        loads = np.array(list(self.hvac_thermal_loads), dtype=float)
        if loads.size > 0 and not np.all(np.isnan(loads)):
            lmin, lmax = np.nanmin(loads), np.nanmax(loads)
            lmargin = max(100.0, (lmax - lmin) * 0.2)
            self.ax2.set_ylim(lmin - lmargin, lmax + lmargin)

        # 3) Power: floor at 0 with a sensible minimum top so a flat 0 is visible
        powers = np.array(list(self.hvac_powers), dtype=float)
        if powers.size > 0 and not np.all(np.isnan(powers)):
            pmax = np.nanmax(powers)
            self.ax3.set_ylim(0.0, max(pmax * 1.2, 1000.0))

        # 4) Supply Air Temperature
        sat_all = np.concatenate([
            np.array(list(self.supply_air_temps_real), dtype=float),
            np.array(list(self.supply_air_temps_setpt), dtype=float)
        ])
        if sat_all.size > 0 and not np.all(np.isnan(sat_all)):
            smin, smax = np.nanmin(sat_all), np.nanmax(sat_all)
            self.ax4.set_ylim(smin - 1.0, smax + 1.0)

        # 5) Supply Air Flow: keep zero visible (no collapse when all zeros)
        saf_all = np.concatenate([
            np.array(list(self.supply_air_flows_real), dtype=float),
            np.array(list(self.supply_air_flows_setpt), dtype=float)
        ])
        if saf_all.size > 0 and not np.all(np.isnan(saf_all)):
            fmax = np.nanmax(saf_all)
            self.ax5.set_ylim(0.0, max(fmax * 1.2, 0.5))  # >= 0.5 m³/s top

    # --------- Teardown ---------
    def stop(self):
        plt.ioff()
        plt.show()
        print(f"✓ Dashboard stopped. Total points: {len(self.timesteps)}")


# Keep the rest of MultiHVACDashboard class unchanged...
class MultiHVACDashboard:
    """Dashboard for monitoring multiple buildings with 4 subplots and GIF export"""

    def __init__(self, building_names, max_points=200, save_gif=True, gif_filename="hvac_animation.gif"):
        self.building_names = building_names
        self.max_points = max_points
        self.num_buildings = len(building_names)
        self.save_gif = save_gif
        self.gif_filename = gif_filename

        # Data storage
        self.timesteps = deque(maxlen=max_points)
        self.building_data = {
            name: {
                'temperature': deque(maxlen=max_points),
                'thermal_load': deque(maxlen=max_points),
                'power': deque(maxlen=max_points),
                'cumulative_energy': deque(maxlen=max_points)
            }
            for name in building_names
        }

        # GIF frames storage
        self.gif_frames = []

        # Color palette - consistent colors for each building
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.num_buildings))
        self.color_map = {name: self.colors[i] for i, name in enumerate(building_names)}

        # Create figure with 4 subplots
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10), dpi=300)
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])

        # Initialize line plots
        self.temp_lines = {}
        for name in building_names:
            line, = self.ax1.plot([], [], linewidth=2.5, label=name,
                                  color=self.color_map[name])
            self.temp_lines[name] = line

        self.power_line, = self.ax2.plot([], [], 'r-', linewidth=3, label='Total Power')

        self._setup_axes()
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, hspace=0.35, wspace=0.3)
        plt.show(block=False)
        plt.pause(0.001)

        print(f"✓ Multi-Building HVAC Dashboard initialized for {self.num_buildings} buildings")
        if save_gif:
            print(f"✓ GIF recording enabled: {gif_filename}")

    def _setup_axes(self):
        """Configure axes"""
        self.ax1.set_title('Individual Space Air Temperature', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Timestep', fontsize=10)
        self.ax1.set_ylabel('Temperature (°C)', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='best', fontsize=9)

        self.ax2.set_title('Aggregated Electric Load', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Timestep', fontsize=10)
        self.ax2.set_ylabel('Power (W)', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='best', fontsize=9)

        self.ax3.set_title('Real-Time Thermal Load Comparison', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Thermal Load (W)', fontsize=10)
        self.ax3.grid(True, alpha=0.3, axis='y')

        self.ax4.set_title('Cumulative Energy Consumption Comparison', fontsize=12, fontweight='bold')
        self.ax4.set_ylabel('Energy (kWh)', fontsize=10)
        self.ax4.grid(True, alpha=0.3, axis='y')

    def add_data_point(self, timestep, building_data_dict, timestep_duration_s=900):
        """Add data point for all buildings"""
        self.timesteps.append(timestep)

        for building_name in self.building_names:
            if building_name in building_data_dict:
                data = building_data_dict[building_name]
                self.building_data[building_name]['temperature'].append(data['temperature'])
                self.building_data[building_name]['thermal_load'].append(data['thermal_load'])
                self.building_data[building_name]['power'].append(data['power'])

                if len(self.building_data[building_name]['cumulative_energy']) == 0:
                    cumulative = data['power'] * timestep_duration_s / 3600 / 1000
                else:
                    previous = list(self.building_data[building_name]['cumulative_energy'])[-1]
                    cumulative = previous + data['power'] * timestep_duration_s / 3600 / 1000
                self.building_data[building_name]['cumulative_energy'].append(cumulative)

        self._update_plots()

        if self.save_gif:
            self._save_frame()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        if len(self.timesteps) % 20 == 0:
            total_power = sum(building_data_dict[name]['power']
                              for name in self.building_names
                              if name in building_data_dict)
            print(f"  Step {timestep}: Total Power={total_power:.1f}W")

    def _update_plots(self):
        """Update all plots"""
        x_data = list(self.timesteps)

        if len(x_data) < 1:
            return

        for building_name in self.building_names:
            temp_data = list(self.building_data[building_name]['temperature'])
            self.temp_lines[building_name].set_data(x_data, temp_data)

        total_power = np.zeros(len(x_data))
        for building_name in self.building_names:
            power_array = np.array(list(self.building_data[building_name]['power']))
            if len(power_array) < len(x_data):
                power_array = np.pad(power_array, (0, len(x_data) - len(power_array)),
                                     constant_values=0)
            total_power += power_array

        self.power_line.set_data(x_data, total_power)

        current_loads = [
            list(self.building_data[name]['thermal_load'])[-1]
            if len(self.building_data[name]['thermal_load']) > 0 else 0
            for name in self.building_names
        ]

        self.ax3.clear()
        bars3 = self.ax3.bar(range(len(self.building_names)), current_loads,
                             color=[self.color_map[name] for name in self.building_names],
                             alpha=0.7, edgecolor='black', linewidth=1.5)
        self.ax3.set_xticks(range(len(self.building_names)))
        self.ax3.set_xticklabels(self.building_names, rotation=45, ha='right', fontsize=9)
        self.ax3.set_ylabel('Thermal Load (W)', fontsize=10)
        self.ax3.set_title('Real-Time Thermal Load Comparison', fontsize=12, fontweight='bold')
        self.ax3.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars3, current_loads):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        cumulative_energies = [
            list(self.building_data[name]['cumulative_energy'])[-1]
            if len(self.building_data[name]['cumulative_energy']) > 0 else 0
            for name in self.building_names
        ]

        self.ax4.clear()
        bars4 = self.ax4.bar(range(len(self.building_names)), cumulative_energies,
                             color=[self.color_map[name] for name in self.building_names],
                             alpha=0.7, edgecolor='black', linewidth=1.5)
        self.ax4.set_xticks(range(len(self.building_names)))
        self.ax4.set_xticklabels(self.building_names, rotation=45, ha='right', fontsize=9)
        self.ax4.set_ylabel('Energy (kWh)', fontsize=10)
        self.ax4.set_title('Cumulative Energy Consumption Comparison', fontsize=12, fontweight='bold')
        self.ax4.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars4, cumulative_energies):
            height = bar.get_height()
            self.ax4.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        if len(x_data) > 1:
            xmin, xmax = min(x_data), max(x_data)
            self.ax1.set_xlim(xmin, xmax)
            self.ax2.set_xlim(xmin, xmax)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()

    def _save_frame(self):
        """Save current plot as a frame for GIF"""
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        self.gif_frames.append(img.copy())
        buf.close()

    def export_gif(self, duration=100):
        """Export saved frames to GIF file"""
        if not self.gif_frames:
            print("⚠ No frames to export")
            return

        print(f"✓ Exporting {len(self.gif_frames)} frames to {self.gif_filename}...")
        self.gif_frames[0].save(
            self.gif_filename,
            save_all=True,
            append_images=self.gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"✓ GIF saved successfully: {self.gif_filename}")

    def stop(self):
        """Stop the dashboard and export GIF"""
        print(f"\n✓ Stopping multi-building dashboard. Total points: {len(self.timesteps)}")

        if len(self.timesteps) > 0:
            total_energy_all = 0
            for building_name in self.building_names:
                temps = list(self.building_data[building_name]['temperature'])
                powers = list(self.building_data[building_name]['power'])
                loads = list(self.building_data[building_name]['thermal_load'])
                energy = list(self.building_data[building_name]['cumulative_energy'])[-1] if \
                self.building_data[building_name]['cumulative_energy'] else 0
                total_energy_all += energy

                print(f"  {building_name}: Avg Temp={np.mean(temps):.1f}°C, "
                      f"Avg Power={np.mean(powers):.1f}W, Avg Load={np.mean(loads):.1f}W, "
                      f"Total Energy={energy:.2f} kWh")

            print(f"\n  Total Energy Consumed (All Buildings): {total_energy_all:.2f} kWh")

        if self.save_gif:
            self.export_gif()

        plt.close(self.fig)


def create_hvac_dashboard(max_points=200, window_title="HVAC System Monitor"):
    """Create a single building HVAC dashboard"""
    return HVACDashboard(max_points=max_points, window_title=window_title)


def create_multi_dashboard(building_names, max_points=200, save_gif=True,
                           gif_filename="hvac_animation.gif"):
    """Create a multi-building dashboard with GIF export"""
    return MultiHVACDashboard(building_names=building_names, max_points=max_points,
                              save_gif=save_gif, gif_filename=gif_filename)