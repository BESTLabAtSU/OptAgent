from bestopt.scripts.data_process import data_process, plot_data
import pandas as pd
import glob
import os

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "../../data/SFH/BLDG")
raw_dir = os.path.join(data_dir, "raw")
csv_save_dir = os.path.join(data_dir, "clean")
plot_save_dir = os.path.join(data_dir, "display")
os.makedirs(csv_save_dir, exist_ok=True)
os.makedirs(plot_save_dir, exist_ok=True)

# Find files
files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))

for n, file in enumerate(files, start=1):
    df = pd.read_csv(file)

    df_format = data_process(
        Tamb=df["temp_outdoor"],
        Tspace=df["temp_zone_0"],
        pHVAC=df["phvac_0"],
        Occ=df["occ_0"],
        Solar=df["solar"],
        Index=df["Time"],
        Tunit="F",
        Punit="watt",
    )

    # Plots
    plot_data(df_format, plot_type="distribution",
              save_path=os.path.join(plot_save_dir, f"distribution_{n}.png"))
    plot_data(df_format, plot_type="daily", day=50,
              save_path=os.path.join(plot_save_dir, f"daily_{n}_50.png"))

    # Add setpoints
    df_format["setpt_cool"] = df["cooling_setpt_0"].values
    df_format["setpt_heat"] = df["heating_setpt_0"].values

    # Save CSV
    base = os.path.splitext(os.path.basename(file))[0]
    out_csv = os.path.join(csv_save_dir, f"{base}.csv")
    df_format.to_csv(out_csv)

    print(f"[{n}/{len(files)}] Saved: {out_csv}")
