"""
Data structure for BESTOpt runtime environment.
Hierarchical organization: Cluster → Domain → System → Component
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Union
from enum import Enum
import numpy as np
from collections import deque

# @TODO set_domain is important to manager cross domain observations, whihc need to follow a clean variable format, I will revise it later
# @Considering to add a more detailed observation structure


# Enumerations

class DomainType(Enum):
    """Physical domains in the building system."""
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    WATER = "water"
    # @TODO add gas in the future


class SystemType(Enum):
    """System types that can span multiple domains."""
    HVAC = "hvac"
    DER = "der"
    BUILDING = "building"
    WATER = "water"
    GRID = "grid"


class ComponentType(Enum):
    """Component types within systems."""
    # HVAC Components
    FAN = "fan"
    COIL = "coil"
    PUMP = "pump"
    CHILLER = "chiller"
    BOILER = "boiler"
    HEAT_PUMP = "heat_pump"
    COOLING_TOWER = "cooling_tower"
    ICE_TANK = "ice_tank"
    # @TODO add more if needed

    # DER Components
    PV = "pv"
    BATTERY = "battery"
    EV = "ev"
    INVERTER = "inverter"

    # Building Components
    THERMAL_ZONE = "thermal_zone"
    ELECTRICAL_ZONE = "electrical_zone"
    WATER_ZONE = "water_zone"

    # Water Components
    WATER_HEATER = "water_heater"
    WATER_TANK = "water_tank"

    # Grid Components
    METER = "meter"
    TRANSFORMER = "transformer"


class OperationMode(Enum):
    """Generic operation modes."""
    OFF = "off"
    ON = "on"
    IDLE = "idle"
    ACTIVE = "active"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"


class HVACMode(Enum):
    """HVAC-specific operation modes."""
    OFF = "off"
    COOLING = "cooling"
    HEATING = "heating"
    VENTILATION = "ventilation"
    AUTO = "auto"
    ECONOMIZER = "economizer"


class DERMode(Enum):
    """DER-specific operation modes."""
    TIME_OF_USE = "tou"
    SELF_CONSUMPTION = "self_consumption"
    ISLANDED = "island"



class PowerFlowMode(Enum):
    """Electrical power flow modes."""
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"
    V2G = "v2g"  # Vehicle-to-grid
    V2B = "v2b"  # Vehicle-to-building


# Base Classes
"""Cluster → Domain → System → Component, I hope this is the last version, 10-07-2025"""
@dataclass
class ComponentState:
    """Base class for all component states."""
    component_id: str
    component_type: ComponentType
    system_id: str
    timestamp: float = 0.0
    is_active: bool = True
    operation_mode: OperationMode = OperationMode.OFF

    # Cross-domain impacts (for components affecting multiple domains)
    domain_impacts: Dict[DomainType, Dict[str, float]] = field(default_factory=dict)

    def get_domain_impact(self, domain: DomainType, metric: str) -> float:
        """Get the component's impact on a specific domain metric."""
        if domain in self.domain_impacts:
            return self.domain_impacts[domain].get(metric, 0.0)
        return 0.0

    def set_domain_impact(self, domain: DomainType, metric: str, value: float):
        """Set the component's impact on a specific domain metric."""
        if domain not in self.domain_impacts:
            self.domain_impacts[domain] = {}
        self.domain_impacts[domain][metric] = value


@dataclass
class SystemState:
    """Base class for system states that can span multiple domains."""
    system_id: str
    system_type: SystemType
    primary_domain: DomainType  # Primary domain this system operates in
    affected_domains: Set[DomainType] = field(default_factory=set)  # All related domains

    components: Dict[str, ComponentState] = field(default_factory=dict)
    timestamp: float = 0.0
    is_active: bool = True

    # Aggregated metrics per domain
    domain_metrics: Dict[DomainType, Dict[str, float]] = field(default_factory=dict)

    def update_domain_metrics(self):
        """Update aggregated metrics for each domain based on components."""
        self.domain_metrics.clear()

        for component in self.components.values():
            for domain, impacts in component.domain_impacts.items():
                if domain not in self.domain_metrics:
                    self.domain_metrics[domain] = {}

                for metric, value in impacts.items():
                    if metric not in self.domain_metrics[domain]:
                        self.domain_metrics[domain][metric] = 0.0
                    self.domain_metrics[domain][metric] += value


@dataclass
class DomainState:
    """Domain-level state aggregating relevant systems."""
    domain_type: DomainType
    systems: Dict[str, SystemState] = field(default_factory=dict)

    # Domain-specific aggregated metrics
    total_power: float = 0.0  # Electrical
    total_heat_flow: float = 0.0  # Thermal
    total_water_flow: float = 0.0  # Water

    # Cross-domain interactions
    interdomain_flows: Dict[DomainType, float] = field(default_factory=dict)

    def update_aggregations(self):
        """Update domain-level aggregations from systems."""
        if self.domain_type == DomainType.ELECTRICAL:
            self.total_power = sum(
                sys.domain_metrics.get(DomainType.ELECTRICAL, {}).get('power', 0.0)
                for sys in self.systems.values()
            )
        elif self.domain_type == DomainType.THERMAL:
            self.total_heat_flow = sum(
                sys.domain_metrics.get(DomainType.THERMAL, {}).get('heat_flow', 0.0)
                for sys in self.systems.values()
            )
        elif self.domain_type == DomainType.WATER:
            self.total_water_flow = sum(
                sys.domain_metrics.get(DomainType.WATER, {}).get('water_flow', 0.0)
                for sys in self.systems.values()
            )


@dataclass
class ClusterState:
    """Top-level cluster state containing all domains."""
    cluster_id: str
    electrical: DomainState = field(default_factory=lambda: DomainState(DomainType.ELECTRICAL))
    thermal: DomainState = field(default_factory=lambda: DomainState(DomainType.THERMAL))
    water: DomainState = field(default_factory=lambda: DomainState(DomainType.WATER))

    timestamp: float = 0.0

    def update_all(self):
        """Update all domain aggregations."""
        self.electrical.update_aggregations()
        self.thermal.update_aggregations()
        self.water.update_aggregations()

    def get_domain(self, domain_type: DomainType) -> DomainState:
        """Get domain state by type."""
        if domain_type == DomainType.ELECTRICAL:
            return self.electrical
        elif domain_type == DomainType.THERMAL:
            return self.thermal
        elif domain_type == DomainType.WATER:
            return self.water
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")


# Component States

# HVAC Components
@dataclass
class FanComponentState(ComponentState):
    """Fan component state."""
    airflow_m3s: float = 0.0
    pressure_rise_pa: float = 0.0
    power_W: float = 0.0
    speed_fraction: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.FAN
        # Fan impacts both thermal (airflow) and electrical (power) domains
        self.set_domain_impact(DomainType.THERMAL, 'airflow', self.airflow_m3s)
        self.set_domain_impact(DomainType.ELECTRICAL, 'power', self.power_W)



@dataclass
class CoilComponentState(ComponentState):
    """Hydronic/air coil (heating/cooling)."""
    airflow_m3s: float = 0.0
    waterflow_m3s: float = 0.0
    air_inlet_temp_c: float = 0.0
    air_outlet_temp_c: float = 0.0
    water_inlet_temp_c: float = 0.0
    water_outlet_temp_c: float = 0.0
    Q_W: float = 0.0
    power_consumption_w: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.COIL
        # self.set_domain_impact(DomainType.ELECTRICAL, 'power_w', self.power_consumption_w)



@dataclass
class ChillerComponentState(ComponentState):
    """Chiller component state."""
    cooling_capacity_w: float = 0.0
    power_W: float = 0.0
    cop: float = 0.0
    chws_temp_c: float = 7.0
    chwr_temp_c: float = 12.0
    water_flow_m3s: float = 0.0
    energy_J_cum: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.CHILLER
        # Chiller impacts thermal (cooling), electrical (power), and water (flow) domains
        self.set_domain_impact(DomainType.ELECTRICAL, 'power', self.power_W)

@dataclass
class HeatPumpComponentState(ComponentState):
    """Heat pump component state (air- or water-source)."""  
    load_flow_m3s: float = 0.0
    load_inlet_temp_C: float = 0.0
    load_outlet_temp_c: float = 0.0
    
    source_flow_m3s: float = 0.0
    source_inlet_temp_C: float = 0.0
    source_outlet_temp_c: float = 0.0
    
    mode: HVACMode = HVACMode.OFF
    thermal_output_W: float = 0.0
    power_W: float = 0.0
    cop: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.HEATPUMP
        self.set_domain_impact(DomainType.ELECTRICAL, 'power_w', self.power_W)

@dataclass
class BoilerComponentState(ComponentState):
    """Boiler component state."""
    thermal_output_w: float = 0.0
    fuel_input_w: float = 0.0
    efficiency: float = 0.85
    hws_temp_c: float = 80.0
    hwr_temp_c: float = 60.0
    water_flow_m3s: float = 0.0
    energy_j_cum: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.BOILER

@dataclass
class CoolingTowerComponentState(ComponentState):
    """Cooling tower (heat rejection)."""
    heat_rejected_w: float = 0.0
    cw_supply_temp_c: float = 0.0
    cw_return_temp_c: float = 0.0
    cw_flow_m3s: float = 0.0
    energy_J_cum: float = 0.0
    fan_power_W: float = 0.0
    pump_power_W: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.COOLING_TOWER


@dataclass
class PumpComponentState(ComponentState):
    """Standalone."""
    waterflow_m3s: float = 0.0
    power_W: float = 0.0
    energy_J_cum: float = 0.0  # accumulated electrical energy [kWh]

    def __post_init__(self):
        self.component_type = ComponentType.PUMP
        self.set_domain_impact(DomainType.ELECTRICAL, 'power_w', self.power_W)


@dataclass
class IceTankComponentState(ComponentState):
    """Ice thermal storage (charge/discharge)."""
    soc: float = 0.0
    q_actual_w: float = 0.0
    energy_j_cum: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.ICE_TANK



# DER Components
@dataclass
class PVComponentState(ComponentState):
    """PV component state."""
    generation_w: float = 0.0
    cell_temperature_c: float = 0.0
    degradation_factor: float = 0.0
    efficiency: float = 0.0
    losses: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.PV
        self.set_domain_impact(DomainType.ELECTRICAL, 'generation', self.generation_w)


@dataclass
class BatteryComponentState(ComponentState):
    """Battery component state."""
    soc: float = 0.5  # State of charge (0-1)
    power_w: float = 0.0  # Positive=charging, Negative=discharging
    capacity_kwh: float = 0.0
    efficiency: float = 0.95
    charge_speed: float = 0.0
    discharge_speed: float = 0.0
    charge_efficiency: float = 0.0
    temperature: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.BATTERY
        self.set_domain_impact(DomainType.ELECTRICAL, 'power', self.power_w)


@dataclass
class EVComponentState(ComponentState):
    """EV component state."""
    soc: float = 0.5  # State of charge (0-1)
    power_w: float = 0.0  # Positive=charging, Negative=discharging
    capacity_wh: float = 0.0
    charge_speed: float = 0.0
    discharge_speed: float = 0.0
    charge_efficiency: float = 0.0
    initially_connected: bool = True
    def __post_init__(self):
        self.component_type = ComponentType.EV
        self.set_domain_impact(DomainType.ELECTRICAL, 'power', self.power_w)


# Building Components
@dataclass
class ThermalZoneComponentState(ComponentState):
    """Thermal zone component state."""
    temperature: float = 0.0
    humidity_pct: float = 50.0
    heat_gain_w: float = 0.0
    occupancy: int = 0
    temperature_buffer: deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=48))

    def __post_init__(self):
        self.component_type = ComponentType.THERMAL_ZONE
        self.set_domain_impact(DomainType.THERMAL, 'heat_gain', self.heat_gain_w)


@dataclass
class ElectricalZoneComponentState(ComponentState):
    """Electrical zone component state."""
    lighting_load_w: float = 0.0
    plug_load_w: float = 0.0
    total_load_w: float = 0.0
    hvac_load_w : float = 0.0
    building_power_w: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.ELECTRICAL_ZONE
        self.total_load_w = self.lighting_load_w + self.plug_load_w
        self.set_domain_impact(DomainType.ELECTRICAL, 'load', self.total_load_w)


@dataclass
class WaterZoneComponentState(ComponentState):
    """Water zone component state."""
    hot_water_demand_w: float = 0.0

    def __post_init__(self):
        self.component_type = ComponentType.WATER_ZONE
        self.set_domain_impact(DomainType.WATER, 'load', self.hot_water_demand_w)


# System States

@dataclass
class HVACSystemState(SystemState):
    """HVAC system state spanning thermal and electrical domains."""

    def __init__(self, system_id: str):
        super().__init__(
            system_id=system_id,
            system_type=SystemType.HVAC,
            primary_domain=DomainType.THERMAL,
            affected_domains={DomainType.THERMAL, DomainType.ELECTRICAL, DomainType.WATER}
        )


@dataclass
class DERSystemState(SystemState):
    """DER system state in electrical domain."""

    def __init__(self, system_id: str):
        super().__init__(
            system_id=system_id,
            system_type=SystemType.DER,
            primary_domain=DomainType.ELECTRICAL,
            affected_domains={DomainType.ELECTRICAL}
        )


@dataclass
class BuildingSystemState(SystemState):
    """Building system state spanning all domains."""

    def __init__(self, system_id: str):
        super().__init__(
            system_id=system_id,
            system_type=SystemType.BUILDING,
            primary_domain=DomainType.THERMAL,
            affected_domains={DomainType.THERMAL, DomainType.ELECTRICAL, DomainType.WATER}
        )

@dataclass
class WaterSystemState(SystemState):
    """Water system state in water domain."""

    def __init__(self, system_id: str):
        super().__init__(
            system_id=system_id,
            system_type=SystemType.WATER,
            primary_domain=DomainType.WATER,
            affected_domains={DomainType.THERMAL}
        )

# Actions

@dataclass
class ComponentAction:
    """Base class for component-level actions."""
    component_id: str
    component_type: str  # ComponentType enum
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'timestamp': self.timestamp
        }


@dataclass
class FanComponentAction(ComponentAction):
    """Fan control action."""
    speed_fraction: float = 0.0  # 0-1
    airflow_setpoint_m3s: Optional[float] = None

    def __post_init__(self):
        self.component_type = "FAN"


@dataclass
class CoilComponentAction(ComponentAction):
    """Coil control action."""
    valve_position: float = 0.0  # 0-1
    water_flow_setpoint_m3s: Optional[float] = None

    def __post_init__(self):
        self.component_type = "COIL"


@dataclass
class ChillerComponentAction(ComponentAction):
    """Chiller control action."""
    enable: bool = False
    chws_temp_setpoint_c: float = 7.0
    capacity_fraction: float = 1.0  # 0-1

    def __post_init__(self):
        self.component_type = "CHILLER"


@dataclass
class PumpComponentAction(ComponentAction):
    """Pump control action."""
    speed_fraction: float = 0.0  # 0-1
    flow_setpoint_m3s: Optional[float] = None
    pressure_setpoint_pa: Optional[float] = None

    def __post_init__(self):
        self.component_type = "PUMP"

@dataclass
class CoolingTowerComponentAction(ComponentAction):
    """CoolingTower control action."""

    def __post_init__(self):
        self.component_type = "COOLING_TOWER"

@dataclass
class HeatPumpComponentAction(ComponentAction):
    """Heat pump control action."""
    enable: bool = False
    chws_temp_setpoint_c: float = 7.0
    hw_temp_setpoint_c: float = 45.0

    def __post_init__(self):
        self.component_type = "HEAT_PUMP"

@dataclass
class BatteryComponentAction(ComponentAction):
    """Battery control action."""
    power_setpoint_w: float = 0.0  # Positive=charge, Negative=discharge
    mode: str = "IDLE"  # CHARGE, DISCHARGE, IDLE

    def __post_init__(self):
        self.component_type = "BATTERY"


@dataclass
class PVComponentAction(ComponentAction):
    """PV control action."""
    curtailment_factor: float = 1.0  # 0-1, fraction of available power to use

    def __post_init__(self):
        self.component_type = "PV"


@dataclass
class EVComponentAction(ComponentAction):
    """EV control action."""
    power_setpoint_w: float = 0.0  # Positive=charge, Negative=discharge
    mode: str = "IDLE"  # CHARGE, DISCHARGE, IDLE, V2G

    def __post_init__(self):
        self.component_type = "EV"

@dataclass
class SystemAction:
    """Base class for system-level actions."""
    system_id: str
    system_type: str  # SystemType enum
    component_actions: Dict[str, ComponentAction] = field(default_factory=dict)
    timestamp: float = 0.0

    def add_component_action(self, component_id: str, action: ComponentAction):
        """Add a component action."""
        self.component_actions[component_id] = action

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system_id': self.system_id,
            'system_type': self.system_type,
            'component_actions': {k: v.to_dict() for k, v in self.component_actions.items()},
            'timestamp': self.timestamp
        }


@dataclass
class HVACSystemAction(SystemAction):
    """HVAC system control action."""
    mode: str = "OFF"  # COOLING, HEATING, VENTILATION, AUTO
    supply_temp_setpoint_c: float = 12.0
    supply_airflow_setpoint_m3s: float = 0.0
    outdoor_airflow_setpoint_m3s: float = 0.2
    cooling_setpoint_c: float = 20
    heating_setpoint_c: float = 20
    hvac_thermal_load: float = 0.0
    hvac_thermal_load_demand: float = 0.0

    def __post_init__(self):
        self.system_type = "HVAC"



@dataclass
class DERSystemAction(SystemAction):
    """DER system control action."""
    mode: str = "TIME_OF_USE"  # SELF_CONSUMPTION, TIME_OF_USE, DEMAND_RESPONSE

    pv2building: float = 0.0
    pv2grid: float = 0.0

    # Dictionary fields for multiple component IDs
    pv2battery: Dict[str, float] = field(default_factory=dict)  # {bat_id: power}
    pv2ev: Dict[str, float] = field(default_factory=dict)  # {ev_id: power}

    battery2building: Dict[str, float] = field(default_factory=dict)
    battery2ev: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {bat_id: {ev_id: power}}

    ev2building: Dict[str, float] = field(default_factory=dict)

    grid2building: float = 0.0
    grid2battery: Dict[str, float] = field(default_factory=dict)
    grid2ev: Dict[str, float] = field(default_factory=dict)


@dataclass
class DomainAction:
    """Domain-level actions coordinating multiple systems."""
    domain_type: str = ""  # DomainType enum
    system_actions: Dict[str, SystemAction] = field(default_factory=dict)

    # Domain-level coordination
    coordination_mode: str = "DECENTRALIZED"  # DECENTRALIZED, CENTRALIZED, HIERARCHICAL

    def add_system_action(self, system_id: str, action: SystemAction):
        """Add a system action."""
        self.system_actions[system_id] = action

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'domain_type': self.domain_type,
            'system_actions': {k: v.to_dict() for k, v in self.system_actions.items()},
            'coordination_mode': self.coordination_mode
        }


@dataclass
class ThermalDomainAction(DomainAction):
    """Thermal domain coordination action."""
    # Domain-wide thermal objectives
    total_cooling_demand_w: Optional[float] = None
    total_heating_demand_w: Optional[float] = None

    # Coordination between HVAC systems (for multi-building)
    load_sharing_strategy: str = "PROPORTIONAL"  # PROPORTIONAL, PRIORITY, OPTIMAL

    def __post_init__(self):
        self.domain_type = "THERMAL"



@dataclass
class ElectricalDomainAction(DomainAction):
    """Electrical domain coordination action."""
    # Domain-wide electrical objectives
    total_demand_limit_w: Optional[float] = None
    renewable_priority: bool = True

    # Inter-system energy transfers (for community DER)
    inter_system_transfers: Dict[tuple, float] = field(default_factory=dict)

    # Format: {(from_system_id, to_system_id): power_w}

    def __post_init__(self):
        self.domain_type = "ELECTRICAL"



@dataclass
class WaterDomainAction(DomainAction):
    """Water domain coordination action."""
    total_flow_limit_m3s: Optional[float] = None

    def __post_init__(self):
        self.domain_type = "WATER"


@dataclass
class ClusterAction:
    """
    Top-level cluster action containing all domain actions.
    This represents the complete control decision for a cluster.
    """
    cluster_id: str
    electrical: ElectricalDomainAction = field(default_factory=ElectricalDomainAction)
    thermal: ThermalDomainAction = field(default_factory=ThermalDomainAction)
    water: WaterDomainAction = field(default_factory=WaterDomainAction)
    timestamp: float = 0.0

    # Cluster-level coordination
    optimization_objective: str = "COST"  # COST, EMISSIONS, COMFORT, BALANCED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cluster_id': self.cluster_id,
            'electrical': self.electrical.to_dict(),
            'thermal': self.thermal.to_dict(),
            'water': self.water.to_dict(),
            'timestamp': self.timestamp,
            'optimization_objective': self.optimization_objective
        }


# Disturbances

@dataclass
class WeatherDisturbance:
    """Weather-related disturbances."""
    outdoor_dry_bulb_temp: float = 20.0
    outdoor_wet_bulb_temp: float = 20.0
    outdoor_humidity_pct: float = 50.0
    solar_radiation_w_m2: float = 0.0
    wind_speed_ms: float = 0.0
    precipitation_mm: float = 0.0
    forecast_outdoor_dry_bulb_temp: List[float] = field(default_factory=list)
    forecast_outdoor_wet_bulb_temp: List[float] = field(default_factory=list)
    forecast_solar_radiation_w_m2: List[float] = field(default_factory=list)


@dataclass
class GridDisturbance:
    """Grid-related disturbances."""
    electricity_price: float = 0.10  # $/kWh
    carbon_intensity: float = 500.0  # gCO2/kWh
    demand_response_signal: float = 0.0
    grid_frequency_hz: float = 60.0
    grid_voltage_pu: float = 1.0

class PriceSignals:
    """Electricity and energy price signals."""
    electricity_price: float = 0.10  # $/kWh
    peaksignal: bool = False
    peak_end: int = 0
    peak_start: int = 0
    demand_charge: float = 15.0  # $/kW
    carbon_intensity: float = 500.0  # gCO2/kWh
    gas_price: Optional[float] = None  # $/therm
    forecast_peaksignal: List[float] = field(default_factory=list)

@dataclass
class OccupancyDisturbance:
    """Occupancy-related disturbances."""
    occupancy_count: int = 0
    occupancy_fraction: float = 0.0
    activity_level: float = 1.0  # Metabolic activity multiplier


@dataclass
class Disturbance:
    """Complete disturbance set (global, shared across cluster)."""
    weather: WeatherDisturbance = field(default_factory=WeatherDisturbance)
    grid: GridDisturbance = field(default_factory=GridDisturbance)
    prices: PriceSignals = field(default_factory=PriceSignals)
    occupancy: OccupancyDisturbance = field(default_factory=OccupancyDisturbance)
    timestamp: float = 0.0


# Observations

@dataclass
class SystemObservation:
    """System-level observation."""
    system_id: str
    system_type: SystemType
    metrics: Dict[str, float] = field(default_factory=dict)
    component_states: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class DomainObservation:
    """Domain-level observation."""
    domain_type: DomainType
    system_observations: Dict[str, SystemObservation] = field(default_factory=dict)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClusterObservation:
    """Cluster-level observation."""
    cluster_id: str
    electrical: DomainObservation = field(default_factory=lambda: DomainObservation(DomainType.ELECTRICAL))
    thermal: DomainObservation = field(default_factory=lambda: DomainObservation(DomainType.THERMAL))
    water: DomainObservation = field(default_factory=lambda: DomainObservation(DomainType.WATER))
    timestamp: float = 0.0

    # Time features
    time_of_day: float = 0.0
    day_of_week: int = 1
    day_of_year: int = 1

    # Forecasts
    weather_forecast: List[WeatherDisturbance] = field(default_factory=list)
    price_forecast: List[float] = field(default_factory=list)
    occupancy_forecast: List[float] = field(default_factory=list)


# Helper Functions


def aggregate_cross_domain_impacts(cluster: ClusterState) -> Dict[str, float]:
    """Calculate cross-domain impacts and energy flows."""
    impacts = {
        'hvac_electrical_demand': 0.0,
        'der_generation': 0.0,
        'building_electrical_load': 0.0,
        'thermal_cooling_demand': 0.0,
        'water_consumption': 0.0,
    }

    # Sum HVAC electrical demand from thermal domain systems
    for sys in cluster.thermal.systems.values():
        if sys.system_type == SystemType.HVAC:
            sys.update_domain_metrics()
            impacts['hvac_electrical_demand'] += sys.domain_metrics.get(
                DomainType.ELECTRICAL, {}
            ).get('power', 0.0)
            impacts['thermal_cooling_demand'] += sys.domain_metrics.get(
                DomainType.THERMAL, {}
            ).get('cooling', 0.0)

    # Sum DER generation
    for sys in cluster.electrical.systems.values():
        if sys.system_type == SystemType.DER:
            sys.update_domain_metrics()
            impacts['der_generation'] += sys.domain_metrics.get(
                DomainType.ELECTRICAL, {}
            ).get('generation', 0.0)

    # Sum building loads
    for sys in cluster.electrical.systems.values():
        if sys.system_type == SystemType.BUILDING:
            sys.update_domain_metrics()
            impacts['building_electrical_load'] += sys.domain_metrics.get(
                DomainType.ELECTRICAL, {}
            ).get('load', 0.0)

    return impacts


def propagate_cross_domain_effects(cluster: ClusterState) -> None:
    """Propagate effects across domains after component updates."""
    # Update all system metrics first
    for domain in [cluster.electrical, cluster.thermal, cluster.water]:
        for system in domain.systems.values():
            system.update_domain_metrics()

    # Update domain aggregations
    cluster.update_all()

    # Handle cross-domain interactions
    # Example: HVAC power consumption affects electrical domain
    hvac_power = 0.0
    for sys in cluster.thermal.systems.values():
        if sys.system_type == SystemType.HVAC:
            hvac_power += sys.domain_metrics.get(DomainType.ELECTRICAL, {}).get('power', 0.0)

    # This power becomes a load in the electrical domain
    cluster.electrical.interdomain_flows[DomainType.THERMAL] = hvac_power
