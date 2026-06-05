import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from PIL import Image
import io


class ElectricalDashboard:
    """Dashboard for monitoring building's electrical system including PV, battery, EVs, and loads"""

    def __init__(self, max_points=200, window_title="Electrical System Monitor"):
        self.max_points = max_points

        # Data storage
        self.timesteps = deque(maxlen=max_points)

        # Generation and storage
        self.pv_generation = deque(maxlen=max_points)
        self.battery_soc = deque(maxlen=max_points)
        self.ev_tesla_soc = deque(maxlen=max_points)
        self.ev_nissan_soc = deque(maxlen=max_points)
        self.peak_signal = deque(maxlen=max_points)

        # Load components
        self.total_load = deque(maxlen=max_points)
        self.hvac_load = deque(maxlen=max_points)
        self.cooking_load = deque(maxlen=max_points)
        self.pc_load = deque(maxlen=max_points)
        self.tv_load = deque(maxlen=max_points)
        self.lighting_load = deque(maxlen=max_points)

        # GIF frames storage
        self.gif_frames = []

        # Create figure with 2 subplots
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8), dpi=100)
        self.fig.canvas.manager.set_window_title(window_title)

        # Create subplots
        gs = self.fig.add_gridspec(2, 1, hspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0])  # PV & SOCs
        self.ax2 = self.fig.add_subplot(gs[1])  # Load disaggregation

        # Create twin axis for SOC on the first subplot
        self.ax1_soc = self.ax1.twinx()

        # Initialize lines for PV and SOCs
        self.pv_line, = self.ax1.plot([], [], 'gold', linewidth=2.5, label='PV Generation')
        self.bat_soc_line, = self.ax1_soc.plot([], [], 'b-', linewidth=2, label='Battery SOC', alpha=0.8)
        self.tesla_soc_line, = self.ax1_soc.plot([], [], 'g-', linewidth=2, label='Tesla EV SOC', alpha=0.8)
        self.nissan_soc_line, = self.ax1_soc.plot([], [], 'r-', linewidth=2, label='Nissan EV SOC', alpha=0.8)

        # Peak signal shading (will be updated dynamically)
        self.peak_collections = []

        self._setup_axes()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        print(f"✓ Electrical Dashboard initialized: {window_title}")

    def _setup_axes(self):
        """Configure all subplot axes"""
        # Top subplot - PV Generation and SOCs
        self.ax1.set_title('PV Generation & Storage SOC', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('PV Generation (kW)', fontsize=10)
        self.ax1.tick_params(axis='y')
        self.ax1.grid(True, alpha=0.3)

        self.ax1_soc.set_ylabel('State of Charge (%)', fontsize=10)
        self.ax1_soc.set_ylim(0, 105)

        # Bottom subplot - Load disaggregation
        self.ax2.set_title('Building Load Disaggregation', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Timestep', fontsize=10)
        self.ax2.set_ylabel('Power (kW)', fontsize=10)
        self.ax2.grid(True, alpha=0.3)

    def add_data_point(self, timestep, pv_generation_kw, battery_soc, ev_tesla_soc, ev_nissan_soc,
                       total_load_kw, hvac_power_kw, cooking_kw, pc_kw, tv_kw, lighting_kw,
                       is_peak=False):
        """Add a data point and update all plots"""
        self.timesteps.append(timestep)

        # Generation and storage
        self.pv_generation.append(pv_generation_kw)
        self.battery_soc.append(battery_soc * 100)  # Convert to percentage
        self.ev_tesla_soc.append(ev_tesla_soc * 100)
        self.ev_nissan_soc.append(ev_nissan_soc * 100)
        self.peak_signal.append(is_peak)

        # Loads
        self.total_load.append(total_load_kw)
        self.hvac_load.append(hvac_power_kw)
        self.cooking_load.append(cooking_kw)
        self.pc_load.append(pc_kw)
        self.tv_load.append(tv_kw)
        self.lighting_load.append(lighting_kw)

        # Update plots
        self._update_plots()

        # Save frame for GIF
        self._save_frame()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _update_plots(self):
        """Update all plots with current data"""
        x_data = list(self.timesteps)

        if len(x_data) < 1:
            return

        # Update PV generation line
        self.pv_line.set_data(x_data, list(self.pv_generation))

        # Update SOC lines
        self.bat_soc_line.set_data(x_data, list(self.battery_soc))
        self.tesla_soc_line.set_data(x_data, list(self.ev_tesla_soc))
        self.nissan_soc_line.set_data(x_data, list(self.ev_nissan_soc))

        # Update peak signal shading
        self._update_peak_shading()

        # Update load disaggregation (stacked area chart)
        self.ax2.clear()

        # Prepare data for stacking
        hvac = np.array(list(self.hvac_load))
        cooking = np.array(list(self.cooking_load))
        pc = np.array(list(self.pc_load))
        tv = np.array(list(self.tv_load))
        lighting = np.array(list(self.lighting_load))

        # Stack the loads
        self.ax2.fill_between(x_data, 0, hvac,
                              alpha=0.7, color='steelblue', label='HVAC')
        self.ax2.fill_between(x_data, hvac, hvac + cooking,
                              alpha=0.7, color='orange', label='Cooking')
        self.ax2.fill_between(x_data, hvac + cooking, hvac + cooking + pc,
                              alpha=0.7, color='green', label='PC')
        self.ax2.fill_between(x_data, hvac + cooking + pc, hvac + cooking + pc + tv,
                              alpha=0.7, color='purple', label='TV')
        self.ax2.fill_between(x_data, hvac + cooking + pc + tv,
                              hvac + cooking + pc + tv + lighting,
                              alpha=0.7, color='yellow', label='Lighting')

        # Add total load line
        self.ax2.plot(x_data, list(self.total_load), 'k-', linewidth=2,
                     label='Total Load', alpha=0.8)

        # Re-setup the bottom axis
        self.ax2.set_title('Building Load Disaggregation', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Timestep', fontsize=10)
        self.ax2.set_ylabel('Power (kW)', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper left', fontsize=9, ncol=3)

        # Autoscale x-axis
        if len(x_data) > 1:
            xmin, xmax = min(x_data), max(x_data)
            self.ax1.set_xlim(xmin, xmax)
            self.ax2.set_xlim(xmin, xmax)

            # Autoscale PV axis
            pv_data = list(self.pv_generation)
            if pv_data:
                pv_max = max(pv_data) * 1.1 if max(pv_data) > 0 else 1
                self.ax1.set_ylim(0, pv_max)

            # Autoscale load axis
            total_data = list(self.total_load)
            if total_data:
                load_max = max(total_data) * 1.1 if max(total_data) > 0 else 1
                self.ax2.set_ylim(0, load_max)

        # Update legends
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax1_soc.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    def _update_peak_shading(self):
        """Update peak signal shading on the PV/SOC plot"""
        # Clear previous peak shading
        for coll in self.peak_collections:
            coll.remove()
        self.peak_collections.clear()

        # Find peak periods
        x_data = list(self.timesteps)
        peak_data = list(self.peak_signal)

        if len(x_data) < 2:
            return

        # Find start and end of peak periods
        in_peak = False
        peak_start = None

        for i, is_peak in enumerate(peak_data):
            if is_peak and not in_peak:
                peak_start = x_data[i]
                in_peak = True
            elif not is_peak and in_peak:
                # End of peak period
                coll = self.ax1.axvspan(peak_start, x_data[i],
                                       alpha=0.2, color='red', label='Peak Period')
                self.peak_collections.append(coll)
                in_peak = False

        # Handle case where peak period extends to the end
        if in_peak and peak_start is not None:
            coll = self.ax1.axvspan(peak_start, x_data[-1],
                                   alpha=0.2, color='red', label='Peak Period')
            self.peak_collections.append(coll)

    def _save_frame(self):
        """Save current plot as a frame for GIF"""
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        self.gif_frames.append(img.copy())
        buf.close()

    def save_as_gif(self, filename='electrical_dashboard.gif', fps=10):
        """Export saved frames to GIF file"""
        if not self.gif_frames:
            print("⚠ No frames to export")
            return

        print(f"✓ Exporting {len(self.gif_frames)} frames to {filename}...")
        self.gif_frames[0].save(
            filename,
            save_all=True,
            append_images=self.gif_frames[1:],
            duration=1000//fps,  # Convert fps to duration in milliseconds
            loop=0
        )
        print(f"✓ GIF saved successfully: {filename}")

    def stop(self):
        """Stop the plotter and show final statistics"""
        print(f"✓ Stopping electrical dashboard. Total points: {len(self.timesteps)}")

        if len(self.timesteps) > 0:
            # Calculate statistics
            avg_pv = np.mean(list(self.pv_generation))
            max_pv = np.max(list(self.pv_generation))
            avg_total_load = np.mean(list(self.total_load))
            max_total_load = np.max(list(self.total_load))

            # SOC statistics
            final_bat_soc = list(self.battery_soc)[-1]
            final_tesla_soc = list(self.ev_tesla_soc)[-1]
            final_nissan_soc = list(self.ev_nissan_soc)[-1]

            print(f"\n:bar_chart: Electrical System Statistics:")
            print(f"  PV Generation: Avg={avg_pv:.2f}kW, Max={max_pv:.2f}kW")
            print(f"  Total Load: Avg={avg_total_load:.2f}kW, Max={max_total_load:.2f}kW")
            print(f"  Final SOCs: Battery={final_bat_soc:.1f}%, Tesla={final_tesla_soc:.1f}%, Nissan={final_nissan_soc:.1f}%")

            # Load breakdown
            avg_hvac = np.mean(list(self.hvac_load))
            avg_cooking = np.mean(list(self.cooking_load))
            avg_pc = np.mean(list(self.pc_load))
            avg_tv = np.mean(list(self.tv_load))
            avg_lighting = np.mean(list(self.lighting_load))

            print(f"\n  Load Breakdown (Average):")
            print(f"    HVAC: {avg_hvac:.2f}kW ({avg_hvac/avg_total_load*100:.1f}%)")
            print(f"    Cooking: {avg_cooking:.2f}kW ({avg_cooking/avg_total_load*100:.1f}%)")
            print(f"    PC: {avg_pc:.2f}kW ({avg_pc/avg_total_load*100:.1f}%)")
            print(f"    TV: {avg_tv:.2f}kW ({avg_tv/avg_total_load*100:.1f}%)")
            print(f"    Lighting: {avg_lighting:.2f}kW ({avg_lighting/avg_total_load*100:.1f}%)")

        plt.ioff()
        plt.show()


def create_electrical_dashboard(max_points=200, window_title="Electrical System Monitor"):
    """Create an electrical system dashboard"""
    return ElectricalDashboard(max_points=max_points, window_title=window_title)