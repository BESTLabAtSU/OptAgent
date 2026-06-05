"""
YAML Configuration Generator for BESTOpt Agents and Tools
Generates all agent and tool YAML files automatically
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class YAMLConfigGenerator:
    """Generate YAML configurations for agents and tools"""

    def __init__(self, config_dir: Path = Path("../config")):
        self.config_dir = config_dir
        self.agents_dir = config_dir / "agents"
        self.tools_dir = config_dir / "tools"

        # Create directories
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_configs(self):
        """Generate all agent and tool configurations"""
        print("Generating agent configurations...")
        self._generate_all_agents()

        # print("Generating tool configurations...")
        # self._generate_all_tools()

        print(f"✅ Generated configs in {self.config_dir}")

    def _generate_all_agents(self):
        """Generate all agent YAML files"""

        agents = [
            {
                "agent_id": "cluster_agent",
                "name": "Cluster Configuration Agent",
                "role": "Configure and query cluster-related settings",
                "description": "Expert in cluster level configurations",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Add a new cluster",
                    "Change the location at 'Syracuse'",
                ],
                "available_tools": [
                    "add_cluster",
                ],
                "example_tasks": [
                    "Add a new cluster named 'Residential_cluster'",
                    "Set the cluster location at 'Syracuse'"
                ],
                "constraints": [

                ]
            },

            {
                "agent_id": "building_agent",
                "name": "Building Configuration Agent",
                "role": "Configure and query building-related settings",
                "description": "Expert in building, thermal zones, and electrical zones configurations",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Add and configure buildings",
                    "Manage thermal and electrical zones",
                    "Update building parameters",
                    "Query building configurations",
                    "Select buildings for simulation"
                ],
                "available_tools": [
                    "building_add",
                    "building_update",
                    "building_remove",
                    "building_query",
                    "building_select",
                    "building_add_thermal_zone",
                    "building_add_electrical_zone"
                ],
                "example_tasks": [
                    "Add a new office building",
                    "Add a LSTM based thermal dynamic model",
                    "Query current building setup"
                ],
                "constraints": [
                    "Validate building IDs are unique",
                    "Ensure cluster exists before adding"
                ]
            },
            {
                "agent_id": "der_agent",
                "name": "DER Systems Agent",
                "role": "Configure, add, update, query distributed energy resources-related settings",
                "description": "Expert in PV systems, battery storage and EVs configuration",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Configure PV solar systems",
                    "Manage battery energy storage",
                    "Setup EV charging parameters",
                    "Update DER settings",
                ],
                "available_tools": [
                    "der_add",
                    "der_update",
                    "der_remove",
                    "der_query",
                    "der_assign_to_buildings",
                    "der_select"
                ],
                "example_tasks": [
                    "Upgrade a 20kWh battery system",
                    "Replace by a 50kW PV array",
                    "Add two EVs"
                ],
                "constraints": ["Ensure power ratings are realistic"]
            },
            {
                "agent_id": "hvac_agent",
                "name": "HVAC Systems Agent",
                "role": "Configure, add, update, query HVAC system settings",
                "description": "Expert in HVAC system configuration",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Configure HVAC systems",
                    "Setup HVAC component such as fan, coil, chiller",
                ],
                "available_tools": [
                    "hvac_add",
                    "hvac_update",
                    "hvac_remove",
                    "hvac_query",
                    "hvac_assign_to_buildings",
                    "hvac_select"
                ],
                "example_tasks": [
                    "Add a FCU system for office building",
                    "Set chiller capacity to 50kW",
                    "Query HVAC efficiency metrics",
                    "Change COP to 4.0"
                ],
                "constraints": ["Match HVAC capacity to building loads"]
            },
            {
                "agent_id": "controller_agent",
                "name": "Control Strategy Agent",
                "role": "Control systems specialist manage controller setup",
                "description": "Expert in detailed controller configuration setup",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Configure control strategies",
                    "Manage rule-based control parameters",
                    "Configure setpoints and pre-cooling",
                    "Manage control parameters"
                ],
                "available_tools": [
                    "controller_add_hvac",
                    "controller_add_der",
                    "controller_update",
                    "controller_remove",
                    "controller_query",
                    "controller_assign_to_system"
                ],
                "example_tasks": [
                    "Use MPC controller for HVAC",
                    "Configure battery charge controller",
                    "Set temperature setpoints to 20C"
                ],
                "constraints": ["Ensure control compatibility with systems"]
            },
            {
                "agent_id": "disturbance_agent",
                "name": "Disturbance Agent",
                "role": "External factors specialist managing weather, occupancy, and pricing",
                "description": "Expert in external factors affecting buildings",
                "model": "gpt-3.5-turbo",
                "temperature": 0.2,
                "capabilities": [
                    "Configure weather data",
                    "Manage occupancy patterns",
                    "Setup electricity pricing",
                    "Configure external disturbances",
                    "Manage temporal variations"
                ],
                "available_tools": [
                    "disturbance_add_weather",
                    "disturbance_add_occupancy",
                    "disturbance_add_price",
                    "disturbance_update",
                    "disturbance_remove",
                    "disturbance_query",
                    "disturbance_select"
                ],
                "example_tasks": [
                    "Load weather data for Phoenix",
                    "Configure office occupancy schedule",
                    "Setup time-of-use pricing"
                ],
                "constraints": ["Ensure time alignment with simulation"]
            },
            {
                "agent_id": "environment_agent",
                "name": "Environment Agent",
                "role": "Simulation environment specialist managing simulation settings",
                "description": "Expert in simulation configuration, time settings, and parameters",
                "model": "gpt-3.5-turbo",
                "temperature": 0.2,
                "capabilities": [
                    "Configure simulation duration",
                    "Set time resolution",
                    "Manage simulation parameters",
                    "Configure start times",
                    "Setup computational settings"
                ],
                "available_tools": [
                    "environment_setup",
                    "environment_update",
                    "environment_query",
                    "environment_select"
                ],
                "example_tasks": [
                    "Set simulation to 24 hours",
                    "Configure 15-minute timesteps",
                    "Set simulation start date"
                ],
                "constraints": ["Respect computational limits"]
            },
            {
                "agent_id": "config_agent",
                "name": "Configuration Manager Agent",
                "role": "High-level configuration agent for file operations and validation",
                "description": "Manages complete configuration lifecycles including creating, saving, validating, and inspecting entire system configurations.",
                "model": "gpt-4",
                "temperature": 0.2,
                "capabilities": [
                    "Load complete configuration files",
                    "Save current setup to configuration files",
                    "Validate configuration completeness and consistency",
                    "List and summarize all configured components",
                ],
                "available_tools": [
                    "config_create",
                    "config_save",
                    "config_validate",
                    "config_set_active",
                    "config_list",
                ],
                "example_tasks": [
                    "Validate if current setup is ready for simulation",
                    "Save current setup as scenario_1",
                    "Show summary of all buildings, DERs, and controllers",
                ],
                "constraints": [
                    "Ensure file name are valid",
                    "Check for required components before saving"
                ]
            },
            {
                "agent_id": "simulation_agent",
                "name": "Simulation Agent",
                "role": "Specialist running and monitoring simulations",
                "description": "Expert in simulation execution, progress monitoring, and result management",
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "capabilities": [
                    "Execute simulations",
                    "Monitor simulation progress",
                    "Manage simulation results",
                    "Handle simulation errors",
                    "Track simulation status"
                ],
                "available_tools": [
                    "simulation_run",
                    "simulation_get_status",
                    "simulation_list_results"
                ],
                "example_tasks": [
                    "Run baseline simulation",
                    "Check simulation progress"
                ],
                "constraints": ["Check configuration validity first"]
            },
            {
                "agent_id": "analysis_agent",
                "name": "Analysis Agent",
                "role": "Data analysis specialist evaluating simulation results",
                "description": "Expert in performance metrics, data analysis, and recommendations",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Analyze comfort metrics",
                    "Calculate energy efficiency",
                    "Evaluate costs and savings",
                    "Assess flexibility metrics",
                    "Generate recommendations"
                ],
                "available_tools": [
                    "analysis_comfort",
                    "analysis_energy",
                    "analysis_cost",
                    "analysis_flexibility",
                    "analysis_comprehensive"
                ],
                "example_tasks": [
                    "Analyze comfort violations",
                    "Calculate energy savings",
                    "Evaluate cost effectiveness"
                ],
                "constraints": ["Ensure data availability"]
            },
            {
                "agent_id": "comparison_agent",
                "name": "Comparison Agent",
                "role": "Scenario comparison specialist evaluating differences between simulations",
                "description": "Expert in multi-scenario analysis and comparison",
                "model": "gpt-4",
                "temperature": 0.3,
                "capabilities": [
                    "Compare simulation results",
                    "Identify improvements",
                    "Rank scenarios",
                    "Generate comparison reports",
                    "Recommend best options"
                ],
                "available_tools": [
                    "comparison_comfort",
                    "comparison_energy",
                    "comparison_comprehensive"
                ],
                "example_tasks": [
                    "Compare baseline with battery scenario",
                    "Identify best configuration",
                    "Rank scenarios by efficiency"
                ],
                "constraints": ["Require at least two simulations"]
            }
        ]

        for agent in agents:
            filepath = self.agents_dir / f"{agent['agent_id']}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(agent, f, default_flow_style=False, sort_keys=False)
            print(f"  ✓ Generated {agent['agent_id']}.yaml")

    def validate_configs(self):
        """Validate all generated configurations"""
        print("\nValidating configurations...")

        # Check agent files
        agent_count = len(list(self.agents_dir.glob("*.yaml")))
        print(f"  ✓ Found {agent_count} agent configurations")

        # Check tool files
        tool_count = len(list(self.tools_dir.glob("*.yaml")))
        print(f"  ✓ Found {tool_count} tool configurations")

        # Validate cross-references
        errors = []

        for agent_file in self.agents_dir.glob("*.yaml"):
            with open(agent_file, 'r') as f:
                agent = yaml.safe_load(f)

            for tool_id in agent.get('available_tools', []):
                tool_file = self.tools_dir / f"{tool_id}.yaml"
                if not tool_file.exists():
                    errors.append(f"Agent {agent['agent_id']} references missing tool: {tool_id}")

        if errors:
            print("  ⚠ Validation errors:")
            for error in errors:
                print(f"    - {error}")
        else:
            print("  ✓ All configurations valid")


def main():
    """Generate all YAML configurations"""
    import sys

    # Determine config directory
    if len(sys.argv) > 1:
        config_dir = Path(sys.argv[1])
    else:
        # Try to find project root
        current = Path.cwd()
        if (current / "src").exists():
            config_dir = current / "config"
        elif current.name == "src":
            config_dir = current.parent / "config"
        else:
            config_dir = current / "config"

    print(f"Generating configurations in: {config_dir}")

    generator = YAMLConfigGenerator(config_dir)
    generator.generate_all_configs()
    generator.validate_configs()

    print(f"\n✅ Complete! Configurations generated in: {config_dir}")
    print("\nTo use in your code:")
    print(f"  config_dir = Path('{config_dir}')")


if __name__ == "__main__":
    main()