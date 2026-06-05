"""
BaseModule for BESTOpt
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging


class BaseModule(ABC):
    """
    Abstract base class for all dynamic modules.

    Every dynamic module (e.g., Building, HVAC, DERs...) should inherit from
    this class and implement initialize(), step(), and reset() methods.

    The step() method is now flexible - subclasses can override with
    whatever parameters they need.
    """

    def __init__(self, config: Dict[str, Any], name: str = "BaseModule"):
        """
        Initialize base module.

        Args:
            config: Module configuration dictionary
            name: Module name, also used for namespacing the logger.
        """
        self.config = config
        self.name = name

        # Module-level logger
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Tracks if the module has been initialized before stepping
        self._initialized = False

        # Optional: store time-series of states if history enabled
        self._state_history: List[Dict[str, Any]] = []
        self._enable_history = config.get('enable_history', False)

    # ------------------- ABSTRACT INTERFACE -------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the module with configuration.

        Example: allocate arrays, set initial conditions, load parameters.
        Must be called before simulation starts.
        """
        pass

    def step(self, *args, **kwargs) -> Any:
        """
        Execute one simulation timestep.

        This is the flexible base implementation that can be overridden
        with any signature needed by the specific module.

        Common patterns:
        - HVAC: step(state, action, disturbance, timestep)
        - DER: step(state, action, disturbance, resolution, timestep)
        - Controller: step(state, observation, disturbance, timestep)
        - Disturbance: step(current_step)
        - ThermalZone: step(state, action, disturbance, timestep)

        Returns:
            Module-specific outputs (dict, state object, or other)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement step() method"
        )

    @abstractmethod
    def reset(self) -> None:
        """
        Reset module to its initial state.

        Ensures reproducibility between episodes or experiments.
        Example: set SOC = 100%.
        """
        pass


    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input parameters.

        Args:
            **kwargs: Arbitrary key-value pairs for inputs.

        Returns:
            True if inputs are valid (default always True).
            Override in child classes for input sanity checks.
        """
        return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get current module state as dictionary.

        Example keys: {"indoor_temp": 22.5, "SOC": 0.8}
        Used for evaluation, saving, and debugging.
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set module state from dictionary.

        Args:
            state: State dictionary to restore.
        This allows checkpointing and restoring simulations.
        """
        pass


    def save_state(self, filepath: str) -> None:
        """Save module state to file using pickle."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, filepath: str) -> None:
        """Load module state from file using pickle and apply to module."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.set_state(state)


    def _record_state(self, state: Dict[str, Any]) -> None:
        """Record state in history if enabled."""
        if self._enable_history:
            self._state_history.append(state.copy())

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get recorded state history.

        Returns:
            List of state dicts from previous timesteps.
        """
        return self._state_history.copy()

    def clear_history(self) -> None:
        """Clear state history."""
        self._state_history.clear()

    def check_initialized(self) -> None:
        """
        Check if module has been initialized.
        Raises RuntimeError if not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} '{self.name}' must be initialized "
                "before calling step()"
            )

    def get_config_value(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def log_debug(self, message: str) -> None:
        self.logger.debug(f"[{self.name}] {message}")

    def log_info(self, message: str) -> None:
        self.logger.info(f"[{self.name}] {message}")

    def log_warning(self, message: str) -> None:
        self.logger.warning(f"[{self.name}] {message}")

    def log_error(self, message: str) -> None:
        self.logger.error(f"[{self.name}] {message}")

