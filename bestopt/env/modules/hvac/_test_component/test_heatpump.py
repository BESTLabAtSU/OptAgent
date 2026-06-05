"""
Unit test for HeatPumpModule (pure unittest, Jupyter-safe)
"""

#%% Imports
import sys, os, unittest
from types import SimpleNamespace
from math import isclose

# --- Ensure project root is visible to Python ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

# --- Import target module ---
from bestopt.env.modules.hvac.component.heat_pump import HeatPumpModule


# --- Minimal stubs for data structures ---
class HVACMode:
    OFF = SimpleNamespace(value="OFF")
    HEATING = SimpleNamespace(value="HEATING")
    COOLING = SimpleNamespace(value="COOLING")


class HeatPumpComponentState:
    def __init__(self):
        self.load_flow_m3s = 0.05
        self.source_flow_m3s = 0.5
        self.load_inlet_temp_C = 35.0
        self.source_inlet_temp_C = 10.0
        self.load_outlet_temp_C = 0.0
        self.source_outlet_temp_C = 0.0
        self.thermal_output_W = 0.0
        self.power_W = 0.0
        self.cop = 0.0
        self.mode = HVACMode.OFF


class HeatPumpComponentAction:
    def __init__(self):
        self.hw_temp_setpoint_c = 45.0
        self.chw_temp_setpoint_c = 7.0


class HVACSystemAction:
    def __init__(self, mode):
        self.mode = mode


# --- Test Suite ---
class TestHeatPumpModule(unittest.TestCase):
    def setUp(self):
        config = {
            "rated_capacity_heating_W": 10000,
            "rated_capacity_cooling_W": 8000,
            "rated_heating_cop": 4.0,
            "rated_cooling_cop": 3.5,
            "rated_load_flow_m3s": 0.05,
            "rated_source_flow_m3s": 0.5,
        }
        self.hp = HeatPumpModule(config)

        # --- Inject temperature-sensitive performance curves ---
        # Heating mode: better performance with warmer source
        self.hp.CAPFT_coeffs_heating = [1.0, 0.0, 0.0, 0.01, 0.0, 0.0]  # +1% per 10°C source rise
        self.hp.EIRFT_coeffs_heating = [1.0, 0.0, 0.0, -0.01, 0.0, 0.0] # -1% per 10°C source rise
        self.hp.EIRFPLR_coeffs_heating = [1.0, 0.0, 0.0]

        # Cooling mode: worse performance with warmer source
        self.hp.CAPFT_coeffs_cooling = [1.0, 0.0, 0.0, -0.01, 0.0, 0.0] # -1% per 10°C source rise
        self.hp.EIRFT_coeffs_cooling = [1.0, 0.0, 0.0, 0.01, 0.0, 0.0]  # +1% per 10°C source rise
        self.hp.EIRFPLR_coeffs_cooling = [1.0, 0.0, 0.0]


    # --- Basic behavior tests ---
    def test_heating_mode(self):
        state = HeatPumpComponentState()
        action = HeatPumpComponentAction()
        sys_action = HVACSystemAction(HVACMode.HEATING)
        result = self.hp.step(state, action, sys_action)

        self.assertEqual(result.mode.value.lower(), "heating")
        self.assertGreater(result.thermal_output_W, 0.0)
        self.assertGreater(result.power_W, 0.0)
        self.assertGreater(result.cop, 0.0)
        self.assertTrue(isclose(result.cop, 4.0, rel_tol=0.5))


    def test_cooling_mode(self):
        state = HeatPumpComponentState()
        state.load_inlet_temp_C = 12.0
        state.source_inlet_temp_C = 35.0
        action = HeatPumpComponentAction()
        sys_action = HVACSystemAction(HVACMode.COOLING)
        result = self.hp.step(state, action, sys_action)

        self.assertEqual(result.mode.value.lower(), "cooling")
        self.assertLess(result.thermal_output_W, 0.0)
        self.assertGreater(result.power_W, 0.0)
        self.assertGreater(result.cop, 0.0)
        self.assertTrue(isclose(result.cop, 3.5, rel_tol=0.5))


    def test_off_mode(self):
        state = HeatPumpComponentState()
        action = HeatPumpComponentAction()
        sys_action = HVACSystemAction(HVACMode.OFF)
        result = self.hp.step(state, action, sys_action)

        self.assertEqual(result.mode.value.lower(), "off")
        self.assertEqual(result.thermal_output_W, 0.0)
        self.assertEqual(result.power_W, 0.0)
        self.assertEqual(result.cop, 0.0)


    def test_reset_and_initialize(self):
        state = HeatPumpComponentState()
        state.mode = HVACMode.HEATING
        state.thermal_output_W = 123.0

        self.hp.reset(state)
        self.assertEqual(state.mode.value.lower(), "off")
        self.assertEqual(state.thermal_output_W, 0.0)

        state.thermal_output_W = 555.0
        self.hp.initialize(state)
        self.assertEqual(state.thermal_output_W, 0.0)


    # --- Performance trend test ---
    def test_cop_vs_source_temp(self):
        """Parametric test: COP should change predictably with source temperature."""
        temps = [0, 10, 20, 30, 40]  # °C source inlet temperatures
        cops_heating, cops_cooling = [], []

        # HEATING MODE
        for T in temps:
            state = HeatPumpComponentState()
            state.source_inlet_temp_C = T
            sys_action = HVACSystemAction(HVACMode.HEATING)
            action = HeatPumpComponentAction()
            result = self.hp.step(state, action, sys_action)
            cops_heating.append(result.cop)

        # COOLING MODE
        for T in temps:
            state = HeatPumpComponentState()
            state.load_inlet_temp_C = 12.0
            state.source_inlet_temp_C = T
            sys_action = HVACSystemAction(HVACMode.COOLING)
            action = HeatPumpComponentAction()
            result = self.hp.step(state, action, sys_action)
            cops_cooling.append(result.cop)

        # Expected trends
        self.assertGreater(cops_heating[-1], cops_heating[0],
            msg=f"COP(40°C)={cops_heating[-1]:.2f} should be higher than COP(0°C)={cops_heating[0]:.2f}")
        self.assertLess(cops_cooling[-1], cops_cooling[0],
            msg=f"COP(40°C)={cops_cooling[-1]:.2f} should be lower than COP(0°C)={cops_cooling[0]:.2f}")

        # Debug printout
        print("\nHeating COP trend vs Source Temp:")
        for T, cop in zip(temps, cops_heating):
            print(f"  Source {T:>2}°C → COP {cop:.3f}")

        print("\nCooling COP trend vs Source Temp:")
        for T, cop in zip(temps, cops_cooling):
            print(f"  Source {T:>2}°C → COP {cop:.3f}")


# --- Entry point (Jupyter/IPython safe) ---
if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
