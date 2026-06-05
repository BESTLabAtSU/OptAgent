import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from typing import Union

def data_process(Tamb: Union[np.ndarray, pd.Series],
                 Tspace: Union[np.ndarray, pd.Series],
                 pHVAC: Union[np.ndarray, pd.Series],
                 Occ: Union[np.ndarray, pd.Series],
                 Solar: Union[np.ndarray, pd.Series],
                 Index: Union[np.ndarray, pd.Series],
                 Tunit: str = "C",
                 Punit: str = "watt",
                 validate_data: bool = True) -> pd.DataFrame:
    """
    Format user data into a structured DataFrame.
    This function takes state, disturbances and control input.
    and formats them into a pandas DataFrame for ModNN dynamic modeling training.
    Time resolution is 15 minutes.
    All input arrays/series will be automatically converted to numpy arrays and reshaped to 1D.
    Temperature data will be converted to Celsius and power data to Watts in the final output.

    Parameters:
    -----------
    Tamb : np.ndarray or pd.Series
        Ambient temperature data - will be converted to numpy array and reshaped to (-1,)
    Tspace : np.ndarray or pd.Series
        Space/room temperature data - will be converted to numpy array and reshaped to (-1,)
    pHVAC : np.ndarray or pd.Series
        HVAC power consumption data - will be converted to numpy array and reshaped to (-1,)
    Occ : np.ndarray or pd.Series
        Occupancy count data (number of people) - will be converted to numpy array and reshaped to (-1,)
    Solar : np.ndarray or pd.Series
        Solar heat gain data - will be converted to numpy array and reshaped to (-1,)
    Index : np.ndarray or pd.Series
        Time index data (datetime strings or datetime objects) - will be converted to numpy array and reshaped to (-1,)
    Tunit : str, optional
        Temperature unit of input data. Supported: "C", "F", "K" (default: "C")
    Punit : str, optional
        Power unit of input data. Supported: "watt", "kw", "btu/h" (default: "watt")
    validate_data : bool, optional
        Whether to perform data validation checks (default: True)

    Returns:
    --------
    pd.DataFrame
        Formatted DataFrame with columns (all standardized units):
        - temp_room: Space temperature in Celsius (°C)
        - temp_amb: Ambient temperature in Celsius (°C)
        - phvac: HVAC power consumption in Watts (W)
        - q_sol: Solar heat gain in Watts (W)
        - occ_num: Occupancy number (dimensionless)
        Index set to datetime format

    Raises:
    -------
    TypeError
        If inputs are not numpy arrays or pandas Series
    ValueError
        If input arrays have different lengths after reshaping, contain invalid data,
        or if unsupported units are provided

    Example:
    --------
    >>> # Create sample data as numpy arrays
    >>> temp_amb = np.array([20.5, 21.0, 21.5, 22.0])  # Celsius
    >>> temp_space = pd.Series([23.0, 23.5, 24.0, 24.5])  # Celsius as pandas Series
    >>> hvac_power = np.array([2.5, 2.8, 3.0, 2.9])  # kW
    >>> occupancy = pd.Series([2, 3, 2, 1])  # Pandas Series
    >>> solar_gain = np.array([500, 600, 700, 650])  # Watts
    >>> timestamps = pd.Series(['2025-01-01 08:00:00', '2025-01-01 09:00:00',
    ...                         '2025-01-01 10:00:00', '2025-01-01 11:00:00'])
    >>>
    >>> df = data_process(temp_amb, temp_space, hvac_power,
    ...                   occupancy, solar_gain, timestamps,
    ...                   Tunit="C", Punit="kw")
    """

    # Helper function to convert pandas Series to numpy array
    def to_numpy_array(data, var_name):
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"{var_name} must be a numpy array or pandas Series. "
                            f"Got {type(data).__name__}. "
                            f"Use np.array() or pd.Series() to convert your data first.")

    # Type checking and conversion - ensure all inputs are numpy arrays or pandas Series
    input_vars = [
        ('Tamb', Tamb), ('Tspace', Tspace), ('pHVAC', pHVAC),
        ('Occ', Occ), ('Solar', Solar), ('Index', Index)
    ]

    # Convert all inputs to numpy arrays
    converted_arrays = {}
    for var_name, var_value in input_vars:
        converted_arrays[var_name] = to_numpy_array(var_value, var_name)

    # Extract converted arrays
    Tamb = converted_arrays['Tamb']
    Tspace = converted_arrays['Tspace']
    pHVAC = converted_arrays['pHVAC']
    Occ = converted_arrays['Occ']
    Solar = converted_arrays['Solar']
    Index = converted_arrays['Index']

    # Validate units
    supported_temp_units = ["C", "F", "K"]
    supported_power_units = ["watt", "kw", "btu/h"]

    if Tunit not in supported_temp_units:
        raise ValueError(f"Unsupported temperature unit '{Tunit}'. "
                         f"Supported units: {supported_temp_units}")

    if Punit not in supported_power_units:
        raise ValueError(f"Unsupported power unit '{Punit}'. "
                         f"Supported units: {supported_power_units}")

    original_shapes = {}

    try:
        original_shapes['Tamb'] = Tamb.shape
        tamb_1d = Tamb.reshape(-1)

        original_shapes['Tspace'] = Tspace.shape
        tspace_1d = Tspace.reshape(-1)

        original_shapes['pHVAC'] = pHVAC.shape
        phvac_1d = pHVAC.reshape(-1)

        original_shapes['Occ'] = Occ.shape
        occ_1d = Occ.reshape(-1)

        original_shapes['Solar'] = Solar.shape
        solar_1d = Solar.reshape(-1)

        original_shapes['Index'] = Index.shape
        index_1d = Index.reshape(-1)

    except Exception as e:
        raise ValueError(f"Error reshaping arrays: {e}")

    # Unit conversion functions
    def convert_temperature_to_celsius(temp_array, from_unit):
        """Convert temperature array to Celsius"""
        if from_unit == "F":
            return (temp_array - 32) * 5 / 9  # Fahrenheit to Celsius
        elif from_unit == "C":
            return temp_array  # Already Celsius
        elif from_unit == "K":
            return (temp_array - 273.15)  # Kelvin to Celsius
        else:
            raise ValueError(f"Unsupported temperature unit: {from_unit}")

    def convert_power_to_watts(power_array, from_unit):
        """Convert power array to Watts"""
        if from_unit == "watt":
            return power_array  # Already in Watts
        elif from_unit == "kw":
            return power_array * 1000  # kW to Watts
        elif from_unit == "btu/h":
            return power_array * 0.293071  # BTU/h to Watts
        else:
            raise ValueError(f"Unsupported power unit: {from_unit}")

    # Convert temperature data to Celsius
    tamb_celsius = convert_temperature_to_celsius(tamb_1d, Tunit)
    tspace_celsius = convert_temperature_to_celsius(tspace_1d, Tunit)

    # Convert power data to Watts
    phvac_watts = convert_power_to_watts(phvac_1d, Punit)
    solar_watts = convert_power_to_watts(solar_1d, Punit)

    # Update the reshaped_arrays dict with converted values for validation
    reshaped_arrays_converted = {
        'Tamb': tamb_celsius, 'Tspace': tspace_celsius, 'pHVAC': phvac_watts,
        'Occ': occ_1d, 'Solar': solar_watts, 'Index': index_1d
    }

    # Data validation checks
    if validate_data:
        # Check if all arrays have the same length after reshaping
        lengths = [len(arr) for arr in reshaped_arrays_converted.values()]

        if len(set(lengths)) > 1:
            raise ValueError(f"All input arrays must have the same length after reshaping. "
                             f"Got lengths: {dict(zip(reshaped_arrays_converted.keys(), lengths))}")

        # Check for empty data
        if lengths[0] == 0:
            raise ValueError("Input arrays cannot be empty")

        # Check for missing values (NaN) and warn user
        missing_counts = {}
        for name, arr in reshaped_arrays_converted.items():
            if arr.dtype.kind in ['f', 'c']:  # float or complex types can have NaN
                missing_counts[name] = np.isnan(arr).sum()
            else:
                missing_counts[name] = 0

        total_missing = sum(missing_counts.values())
        if total_missing > 0:
            print(f"\nWarning: Found {total_missing} missing values (NaN) across all columns:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count} missing values")

    # Create the DataFrame using converted values (all in standard units)
    dynamic_df = pd.DataFrame({
        'temp_room': tspace_celsius,  # °C
        'temp_amb': tamb_celsius,  # °C
        'phvac': phvac_watts,  # Watts
        'solar': solar_watts,  # Watts
        'occ': occ_1d  # dimensionless
    })

    # Convert index to datetime format
    try:
        datetime_index = pd.to_datetime(index_1d, format="%Y-%m-%d %H:%M:%S", errors='coerce')

        # Check if datetime conversion was successful
        if datetime_index.isna().any():
            print("\nWarning: Some datetime values could not be parsed. Trying flexible parsing...")
            datetime_index = pd.to_datetime(index_1d, errors='coerce')

            if datetime_index.isna().any():
                raise ValueError("Unable to parse datetime index. Please check format.")

        dynamic_df.index = datetime_index

    except Exception as e:
        print(f"\nWarning: Could not set datetime index: {e}")
        print("Using default integer index instead.")
        pass

    # Print summary information
    print(f"DATA PROCESSING COMPLETED!")
    print(f"Data Summary:")
    print(dynamic_df.describe())

    return dynamic_df


def plot_data(df: pd.DataFrame,
              plot_type: str = "distribution",
              day: Optional[int] = None,
              figsize: tuple = (15, 10),
              save_path: Optional[str] = None) -> None:
    """
    Plot processed data with different visualization options.

    Parameters:
    -----------
    df : pd.DataFrame
        Processed DataFrame from data_process() function with datetime index
    plot_type : str, optional
        Type of plot to create. Options:
        - "distribution": Show distribution of each column (default)
        - "daily": Show time series for a specific day
    day : int, optional
        Day number to plot (1-based indexing). Required when plot_type="daily"
        Day 1 = first day in dataset, Day 2 = second day, etc.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (15, 10))
    save_path : str, optional
        Path to save the plot. If None, plot is displayed but not saved

    Returns:
    --------
    None
        Displays the plot and optionally saves it

    Raises:
    -------
    ValueError
        If plot_type is not supported or if day is invalid

    Example:
    --------
    >>> # Plot distributions
    >>> plot_data(df, plot_type="distribution")
    >>>
    >>> # Plot day 15 time series
    >>> plot_data(df, plot_type="daily", day=15)
    >>>
    >>> # Save plot to file
    >>> plot_data(df, plot_type="daily", day=5, save_path="day5_plot.png")
    """

    # Validate inputs
    supported_plot_types = ["distribution", "daily"]
    if plot_type not in supported_plot_types:
        raise ValueError(f"Unsupported plot_type '{plot_type}'. "
                         f"Supported types: {supported_plot_types}")

    if plot_type == "daily" and day is None:
        raise ValueError("Day number must be specified when plot_type='daily'")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    if plot_type == "distribution":
        # Create distribution plots for each column
        _plot_distributions(df, figsize, save_path)

    elif plot_type == "daily":
        # Create daily time series subplot
        _plot_daily_timeseries(df, day, figsize, save_path)


def _plot_distributions(df: pd.DataFrame, figsize: tuple, save_path: Optional[str]) -> None:
    """Create distribution plots for each column in the DataFrame."""

    # Get numeric columns only (exclude datetime index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = len(numeric_cols)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Distribution of Dynamic Modeling Variables', fontsize=16, fontweight='bold')

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Define colors and labels for each variable
    plot_info = {
        'temp_room': {'color': 'red', 'label': 'Room Temperature (°C)', 'bins': 30},
        'temp_amb': {'color': 'blue', 'label': 'Ambient Temperature (°C)', 'bins': 30},
        'phvac': {'color': 'green', 'label': 'HVAC Power (W)', 'bins': 25},
        'solar': {'color': 'orange', 'label': 'Solar Radiation (W)', 'bins': 25},
        'occ': {'color': 'purple', 'label': 'Occupancy Count', 'bins': None}
    }


    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        # Get plot information
        info = plot_info.get(col, {'color': 'gray', 'label': col, 'bins': 30})

        if col == 'occ_num':
            # For occupancy (discrete), use bar plot
            occ_counts = df[col].value_counts().sort_index()
            ax.bar(occ_counts.index, occ_counts.values, color=info['color'], alpha=0.7)
            ax.set_xlabel('Number of Occupants')
            ax.set_ylabel('Frequency')
        else:
            # For continuous variables, use histogram with KDE
            ax.hist(df[col].dropna(), bins=info['bins'], color=info['color'],
                    alpha=0.7, density=True, edgecolor='black', linewidth=0.5)

            # Add KDE curve
            try:
                df[col].dropna().plot.kde(ax=ax, color='darkred', linewidth=2)
            except:
                pass  # Skip KDE if it fails

            ax.set_xlabel(info['label'])
            ax.set_ylabel('Density')

        ax.set_title(f'{info["label"]}', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Mean: {df[col].mean():.2f}\nStd: {df[col].std():.2f}\nCount: {df[col].count()}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused subplot
    if n_cols < 6:
        axes[5].set_visible(False)

    # Add overall statistics
    total_points = len(df)
    total_days = total_points / 96  # 96 points per day
    date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"

    fig.text(0.02, 0.02, f'Dataset: {total_points} points ({total_days:.1f} days) | {date_range}',
             fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")

    plt.show()


def _plot_daily_timeseries(df: pd.DataFrame, day: int, figsize: tuple, save_path: Optional[str]) -> None:
    """Create time series plots for a specific day."""

    # Calculate total number of days in dataset
    total_points = len(df)
    total_days = int(total_points / 96)

    if day < 1 or day > total_days:
        raise ValueError(f"Day {day} is out of range. Dataset contains {total_days} days (1-{total_days})")

    # Extract data for the specific day (96 points per day)
    start_idx = (day - 1) * 96
    end_idx = day * 96

    if end_idx > len(df):
        end_idx = len(df)
        print(f"Warning: Day {day} has incomplete data ({end_idx - start_idx} points instead of 96)")

    day_data = df.iloc[start_idx:end_idx].copy()

    # Create time axis (15-minute intervals)
    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
    time_labels = time_labels[:len(day_data)]  # Trim if incomplete day

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Day {day} Time Series - {day_data.index[0].strftime("%Y-%m-%d")}',
                 fontsize=16, fontweight='bold')

    # Define plot configurations
    plot_configs = [
        {
            'ax': axes[0, 0],
            'data': [day_data['temp_room'], day_data['temp_amb']],
            'labels': ['Room Temperature', 'Ambient Temperature'],
            'colors': ['red', 'blue'],
            'ylabel': 'Temperature (°C)',
            'title': 'Temperature Profile'
        },
        {
            'ax': axes[0, 1],
            'data': [day_data['phvac']],
            'labels': ['HVAC Power'],
            'colors': ['green'],
            'ylabel': 'Power (W)',
            'title': 'HVAC Power Consumption'
        },
        {
            'ax': axes[1, 0],
            'data': [day_data['solar']],
            'labels': ['Solar Radiation'],
            'colors': ['orange'],
            'ylabel': 'Power (W)',
            'title': 'Solar Radiation'
        },
        {
            'ax': axes[1, 1],
            'data': [day_data['occ']],
            'labels': ['Occupancy'],
            'colors': ['purple'],
            'ylabel': 'Number of People',
            'title': 'Occupancy Pattern'
        }
    ]

    # Create each subplot
    for config in plot_configs:
        ax = config['ax']

        for i, (data_series, label, color) in enumerate(zip(config['data'], config['labels'], config['colors'])):
            if config['title'] == 'Occupancy Pattern':
                # Step plot for occupancy
                ax.step(range(len(data_series)), data_series.values, where='post',
                        color=color, linewidth=2, label=label)
                ax.fill_between(range(len(data_series)), data_series.values,
                                step='post', alpha=0.3, color=color)
            else:
                # Line plot for continuous variables
                ax.plot(range(len(data_series)), data_series.values,
                        color=color, linewidth=2, label=label, marker='o', markersize=2)

        ax.set_xlabel('Time')
        ax.set_ylabel(config['ylabel'])
        ax.set_title(config['title'], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set x-axis labels (every 2 hours)
        tick_positions = range(0, len(time_labels), 8)  # Every 8 points = 2 hours
        tick_labels = [time_labels[i] for i in tick_positions if i < len(time_labels)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

    # Add summary statistics for the day
    summary_text = (
        f"Day {day} Summary:\n"
        f"Avg Room Temp: {day_data['temp_room'].mean():.1f}°C\n"
        f"Avg HVAC Power: {day_data['phvac'].mean():.0f}W\n"
        f"Max Solar Radiation: {day_data['solar'].max():.0f}W\n"
        f"Avg Occupancy: {day_data['occ'].mean():.1f}"
    )

    fig.text(0.3, 0.3, summary_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Daily plot saved to: {save_path}")

    plt.show()
