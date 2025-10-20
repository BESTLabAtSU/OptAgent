"""
Validation utilities
"""
from typing import Dict, Any, List, Optional


def validate_der_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate DER configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required fields
    required = ["system_id", "parameters"]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate parameters
    if "parameters" in config:
        params = config["parameters"]
        if "system_config" in params:
            system_config = params["system_config"]

            # Validate component configs
            for component in ["pv", "bat", "ev"]:
                if component in system_config:
                    comp_errors = validate_component_config(
                        component,
                        system_config[component]
                    )
                    errors.extend(comp_errors)

    return len(errors) == 0, errors


def validate_component_config(component: str, config: Dict[str, Any]) -> List[str]:
    """Validate individual component configuration"""
    errors = []

    if component == "pv":
        if "rated_capacity_kW" in config:
            if config["rated_capacity_kW"] <= 0:
                errors.append("PV capacity must be positive")

    elif component in ["bat", "ev"]:
        if "rated_capacity_kWh" in config:
            if config["rated_capacity_kWh"] <= 0:
                errors.append(f"{component} capacity must be positive")

        if "initial_soc" in config:
            soc = config["initial_soc"]
            if soc < 0 or soc > 1:
                errors.append(f"{component} SOC must be between 0 and 1")

    return errors