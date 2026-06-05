# ------------------------------------------------------------
# HVAC Single-Building Dashboard (with Water Flow & Chiller SWT)
# ------------------------------------------------------------
import matplotlib
# matplotlib.use("QtAgg")  # Uncomment for interactive display
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

        # ---------------- Data buffers ----------------
        self.timesteps = deque(maxlen=max_points)

        # Existing series
        self.temperatures = deque(maxlen=max_points)
        self.hvac_thermal_loads = deque(maxlen=max_points)
        self.hvac_powers = deque(maxlen=max_points)
        self.cooling_setpoints = deque(maxlen=max_points)
        self.heating_setpoints = deque(maxlen=max_points)
        self.supply_air_temps_real = deque(maxlen=max_points)
        self.supply_air_temps_setpt = deque(maxlen=max_points)
        self.supply_air_flows_real = deque(maxlen=max_points)
        self.supply_air_flows_setpt = deque(maxlen=max_points)

        # New series
        self.water_flow_real = deque(maxlen=max_points)              # Water flow rate (e.g., m³/s)
        self.chiller_supply_water_temps = deque(maxlen=max_points)    # Chilled water supply temp (°C)

        # ---------------- Figure & axes ----------------
        plt.ion()
        # Taller figure to fit 4 rows
        self.fig = plt.figure(figsize=(8.2, 8.8), dpi=120)
        try:
            self.fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

        # 4 rows × 2 cols
        gs = self.fig.add_gridspec(4, 2, hspace=0.85, wspace=0.45)

        self.ax1 = self.fig.add_subplot(gs[0, :])   # Zone temp + setpoints
        self.ax2 = self.fig.add_subplot(gs[1, 0])   # Thermal load
        self.ax3 = self.fig.add_subplot(gs[1, 1])   # HVAC power
        self.ax4 = self.fig.add_subplot(gs[2, 0])   # SAT
        self.ax5 = self.fig.add_subplot(gs[2, 1])   # SAF
        self.ax6 = self.fig.add_subplot(gs[3, 0])   # Water flow
        self.ax7 = self.fig.add_subplot(gs[3, 1])   # Chiller supply water temp

        # ---------------- Lines ----------------
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

        self.saf_real_line, = self.ax5.plot([], [], color="#56B4E9", lw=2, label="Actual Air Flow")
        self.saf_setpt_line, = self.ax5.plot([], [], "--", color="#56B4E9", lw=1.5,
                                             label="Setpoint Air Flow", alpha=0.85, drawstyle="steps-post")

        # New: water flow (single real series)
        self.wf_real_line, = self.ax6.plot([], [], color="#1B9E77", lw=2, label="Water Flow")

        # New: chiller supply water temp (single real series)
        self.swt_line, = self.ax7.plot([], [], color="#D95F02", lw=2, label="Chiller Supply Water Temp")

        self._setup_axes()
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.07, hspace=0.9, wspace=0.45)
        plt.show(block=False)

    # ---------------- Appearance ----------------
    def _setup_axes(self):
        for ax, title, ylabel in [
            (self.ax1, "Zone Temperature & Setpoints", "Temperature (°C)"),
            (self.ax2, "HVAC Thermal Load", "Thermal Load (W)"),
            (self.ax3, "HVAC Power Consumption", "Power (W)"),
            (self.ax4, "Supply Air Temperature", "Temperature (°C)"),
            (self.ax5, "Supply Air Flow Rate", "Flow Rate (m³/s)"),
            (self.ax6, "Water Flow Rate", "Flow Rate (m³/s)"),
            (self.ax7, "Chiller Supply Water Temperature", "Temperature (°C)"),
        ]:
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="gray", alpha=0.25, linestyle="--")
            ax.set_facecolor("#f7f7f7")
            ax.legend(loc="upper right")

        for ax in [self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7]:
            ax.set_xlabel("Timestep")

    # ---------------- Data ingest + redraw ----------------
    def add_data_point(self, timestep, zone_temperature, hvac_thermal_load, hvac_power,
                       supervisory_cooling_setpoint=None, supervisory_heating_setpoint=None,
                       supply_air_temp_real=None, supply_air_temp_setpt=None,
                       supply_air_flow_real=None, supply_air_flow_setpt=None,
                       water_flow_real=None, chiller_supply_water_temp=None):
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

        # New series
        self.water_flow_real.append(_to_float_or_nan(water_flow_real))
        self.chiller_supply_water_temps.append(_to_float_or_nan(chiller_supply_water_temp))

        # Update lines
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
        self.wf_real_line.set_data(x, self.water_flow_real)
        self.swt_line.set_data(x, self.chiller_supply_water_temps)

        if len(x) > 2:
            self._autoscale_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0001)

    # ---------------- Stable autoscaling ----------------
    def _autoscale_axes(self):
        x = list(self.timesteps)
        xmin, xmax = min(x), max(x)
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7]:
            ax.set_xlim(xmin, xmax)

        # Zone temperature (temps + setpoints)
        t_all = np.array(self.temperatures, float)
        cool = np.array(self.cooling_setpoints, float)
        heat = np.array(self.heating_setpoints, float)
        tstack = np.concatenate([t_all, cool, heat])
        if not np.all(np.isnan(tstack)):
            tmin, tmax = np.nanmin(tstack), np.nanmax(tstack)
            self.ax1.set_ylim(tmin - 1.0, tmax + 1.0)

        # Thermal load
        loads = np.array(self.hvac_thermal_loads, float)
        if loads.size and not np.all(np.isnan(loads)):
            lmin, lmax = np.nanmin(loads), np.nanmax(loads)
            self.ax2.set_ylim(lmin - max(100.0, (lmax - lmin) * 0.2),
                              lmax + max(100.0, (lmax - lmin) * 0.2))

        # Power
        powers = np.array(self.hvac_powers, float)
        if powers.size and not np.all(np.isnan(powers)):
            pmax = np.nanmax(powers)
            self.ax3.set_ylim(0.0, max(pmax * 1.2, 1000.0))

        # Supply Air Temperature
        sat_all = np.concatenate([
            np.array(self.supply_air_temps_real, float),
            np.array(self.supply_air_temps_setpt, float)
        ])
        if sat_all.size and not np.all(np.isnan(sat_all)):
            smin, smax = np.nanmin(sat_all), np.nanmax(sat_all)
            self.ax4.set_ylim(smin - 1.0, smax + 1.0)

        # Supply Air Flow
        saf_all = np.concatenate([
            np.array(self.supply_air_flows_real, float),
            np.array(self.supply_air_flows_setpt, float)
        ])
        if saf_all.size and not np.all(np.isnan(saf_all)):
            fmax = np.nanmax(saf_all)
            self.ax5.set_ylim(0.0, max(fmax * 1.2, 0.5))  # >= 0.5 m³/s top

        # Water Flow (new) – same rule as air flow
        wf_all = np.array(self.water_flow_real, float)
        if wf_all.size and not np.all(np.isnan(wf_all)):
            wfmax = np.nanmax(wf_all)
            self.ax6.set_ylim(0.0, max(wfmax * 1.2, 0.005))  # >= 0.05 m³/s top

        # Chiller Supply Water Temperature (new)
        swt_all = np.array(self.chiller_supply_water_temps, float)
        if swt_all.size and not np.all(np.isnan(swt_all)):
            wmin, wmax = np.nanmin(swt_all), np.nanmax(swt_all)
            self.ax7.set_ylim(wmin - 1.0, wmax + 1.0)

    # ---------------- Teardown ----------------
    def stop(self):
        plt.ioff()
        plt.show()
        print(f"✓ Dashboard stopped. Total points: {len(self.timesteps)}")


# Factory
def create_hvac_dashboard(max_points=200, window_title="HVAC System Monitor"):
    return HVACDashboard(max_points=max_points, window_title=window_title)
