"""
MCP Server for Building Simulation Environment
This server exposes building simulation functions as MCP tools that can be called by LLM agents.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import pickle
import numpy as np

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Import your simulation manager
# You'll need to adjust this import based on your project structure
from simulation_manager import SimulationManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BESTOptServer:
    """MCP Server for building simulation operations"""

    def __init__(self, base_path: str = "./simulation_results"):
        """Initialize the MCP server with simulation manager"""
        self.server = Server("building-simulation-server")
        self.sim_manager = SimulationManager(base_path)
        self.active_env = None  # Store active environment
        self.setup_tools()
        self.setup_handlers()

    def setup_tools(self):
        """Define available tools for the MCP server"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available simulation tools"""
            return [
                Tool(
                    name="initialize_environment",
                    description="Initialize the building simulation environment with a configuration file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "config_path": {
                                "type": "string",
                                "description": "Path to the configuration JSON file"
                            },
                            "project_root": {
                                "type": "string",
                                "description": "Project root path (optional)"
                            }
                        },
                        "required": ["config_path"]
                    }
                ),
                Tool(
                    name="run_simulation",
                    description="Run a building simulation and save the results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "total_steps": {
                                "type": "integer",
                                "description": "Number of simulation steps (default 96 for one day)",
                                "default": 96
                            },
                            "simulation_name": {
                                "type": "string",
                                "description": "Name for this simulation run"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the simulation configuration"
                            }
                        },
                        "required": ["simulation_name"]
                    }
                ),
                Tool(
                    name="analyze_simulation",
                    description="Analyze a saved simulation and get comprehensive metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_name": {
                                "type": "string",
                                "description": "Name of the simulation to analyze"
                            }
                        },
                        "required": ["simulation_name"]
                    }
                ),
                Tool(
                    name="compare_simulations",
                    description="Compare two simulations and identify differences",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sim_name1": {
                                "type": "string",
                                "description": "Name of first simulation"
                            },
                            "sim_name2": {
                                "type": "string",
                                "description": "Name of second simulation"
                            },
                            "comparison_name": {
                                "type": "string",
                                "description": "Name for the comparison output (optional)"
                            }
                        },
                        "required": ["sim_name1", "sim_name2"]
                    }
                ),
                Tool(
                    name="list_simulations",
                    description="List all available saved simulations",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_simulation_data",
                    description="Get raw data from a saved simulation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_name": {
                                "type": "string",
                                "description": "Name of the simulation"
                            },
                            "data_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific data keys to retrieve (optional, returns all if not specified)"
                            }
                        },
                        "required": ["simulation_name"]
                    }
                ),
                Tool(
                    name="plot_simulation",
                    description="Generate visualization plots for a simulation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_name": {
                                "type": "string",
                                "description": "Name of the simulation to plot"
                            },
                            "save_figure": {
                                "type": "boolean",
                                "description": "Whether to save the figure",
                                "default": True
                            }
                        },
                        "required": ["simulation_name"]
                    }
                ),
                Tool(
                    name="get_environment_status",
                    description="Get current status of the simulation environment",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

    def setup_handlers(self):
        """Set up tool call handlers"""

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""

            try:
                if name == "initialize_environment":
                    result = await self.initialize_environment(arguments)
                elif name == "run_simulation":
                    result = await self.run_simulation(arguments)
                elif name == "analyze_simulation":
                    result = await self.analyze_simulation(arguments)
                elif name == "compare_simulations":
                    result = await self.compare_simulations(arguments)
                elif name == "list_simulations":
                    result = await self.list_simulations()
                elif name == "get_simulation_data":
                    result = await self.get_simulation_data(arguments)
                elif name == "plot_simulation":
                    result = await self.plot_simulation(arguments)
                elif name == "get_environment_status":
                    result = await self.get_environment_status()
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]

            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e), "tool": name}, indent=2)
                )]

    async def initialize_environment(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the building simulation environment"""
        try:
            from bestopt.env.core.config_manager import ConfigurationManager
            from bestopt.env.core.environment import BESTOptEnvironment

            config_path = arguments["config_path"]
            project_root = arguments.get("project_root", os.path.dirname(os.path.abspath(__file__)))

            # Resolve full path
            if not os.path.isabs(config_path):
                config_path = os.path.join(project_root, config_path)

            # Load configuration
            cm = ConfigurationManager(config_path)
            self.active_env = BESTOptEnvironment(cm.config)

            return {
                "status": "success",
                "message": "Environment initialized successfully",
                "config_path": config_path,
                "total_steps_available": self.active_env.total_step
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def run_simulation(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simulation and save results"""
        try:
            if self.active_env is None:
                return {
                    "status": "error",
                    "message": "Environment not initialized. Please call initialize_environment first."
                }

            total_steps = arguments.get("total_steps", 96)
            simulation_name = arguments["simulation_name"]
            description = arguments.get("description", "")

            config_info = {
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "total_steps": total_steps
            }

            # Run simulation synchronously in executor to avoid blocking
            loop = asyncio.get_event_loop()
            data_dict, save_path = await loop.run_in_executor(
                None,
                self.sim_manager.run_and_save_simulation,
                self.active_env,
                total_steps,
                simulation_name,
                config_info
            )

            # Calculate summary statistics
            summary = {
                "avg_temperature": float(np.mean(data_dict['zone_temperature'])),
                "total_energy_consumed": float(np.sum(data_dict['total_building_load']) * 0.25),
                "pv_generation_total": float(np.sum(data_dict['pv_generation']) * 0.25),
                "battery_cycles": float(self._calculate_battery_cycles(data_dict['battery_soc']))
            }

            return {
                "status": "success",
                "simulation_name": simulation_name,
                "save_path": str(save_path),
                "total_steps": len(data_dict['timesteps']),
                "summary": summary
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def analyze_simulation(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a saved simulation"""
        try:
            simulation_name = arguments["simulation_name"]

            # Run analysis
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None,
                self.sim_manager.analyze_simulation,
                simulation_name
            )

            # Convert numpy types for JSON serialization
            analysis = self._convert_to_serializable(analysis)

            return {
                "status": "success",
                "simulation_name": simulation_name,
                "analysis": analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def compare_simulations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two simulations"""
        try:
            sim_name1 = arguments["sim_name1"]
            sim_name2 = arguments["sim_name2"]
            comparison_name = arguments.get("comparison_name")

            # Run comparison
            loop = asyncio.get_event_loop()
            comparison = await loop.run_in_executor(
                None,
                self.sim_manager.compare_simulations,
                sim_name1,
                sim_name2,
                comparison_name
            )

            # Extract key insights
            metrics = comparison['metrics_comparison']

            return {
                "status": "success",
                "simulation_1": sim_name1,
                "simulation_2": sim_name2,
                "key_differences": {
                    "comfort_violation_diff": metrics['comfort']['violation_rate_diff'],
                    "cost_difference": metrics['cost']['net_cost_diff'],
                    "cost_change_percent": metrics['cost']['net_cost_change_pct'],
                    "renewable_fraction_diff": metrics['efficiency']['renewable_fraction_diff'],
                    "grid_dependency_diff": metrics['efficiency']['grid_dependency_diff']
                },
                "recommendation": self._generate_recommendation(metrics)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def list_simulations(self) -> Dict[str, Any]:
        """List all available simulations"""
        try:
            sim_path = Path(self.sim_manager.base_path)
            simulations = []

            for pkl_file in sim_path.glob("*.pkl"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                        metadata = data.get('metadata', {})
                        simulations.append({
                            "name": pkl_file.stem,
                            "timestamp": metadata.get('timestamp', 'Unknown'),
                            "total_steps": metadata.get('total_steps', 0),
                            "description": metadata.get('config_info', {}).get('description', '')
                        })
                except:
                    continue

            return {
                "status": "success",
                "count": len(simulations),
                "simulations": simulations
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_simulation_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get raw data from a simulation"""
        try:
            simulation_name = arguments["simulation_name"]
            data_keys = arguments.get("data_keys")

            data_dict, metadata = self.sim_manager.load_simulation_data(simulation_name)

            if data_keys:
                filtered_data = {k: data_dict[k].tolist() if isinstance(data_dict[k], np.ndarray)
                else data_dict[k] for k in data_keys if k in data_dict}
            else:
                filtered_data = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                 for k, v in data_dict.items()}

            return {
                "status": "success",
                "simulation_name": simulation_name,
                "metadata": metadata,
                "data": filtered_data
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def plot_simulation(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plots for a simulation"""
        try:
            simulation_name = arguments["simulation_name"]
            save_figure = arguments.get("save_figure", True)

            # Generate plots
            loop = asyncio.get_event_loop()
            fig = await loop.run_in_executor(
                None,
                self.sim_manager.plot_simulation_results,
                simulation_name,
                None,
                save_figure,
                False  # Don't show figure in server mode
            )

            plot_path = self.sim_manager.base_path / f"{simulation_name}_plots.png"

            return {
                "status": "success",
                "simulation_name": simulation_name,
                "plot_saved": save_figure,
                "plot_path": str(plot_path) if save_figure else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status"""
        return {
            "status": "success",
            "environment_initialized": self.active_env is not None,
            "simulation_base_path": str(self.sim_manager.base_path),
            "available_simulations": len(list(Path(self.sim_manager.base_path).glob("*.pkl")))
        }

    def _calculate_battery_cycles(self, soc: np.ndarray) -> float:
        """Calculate equivalent full cycles from SOC data"""
        if len(soc) < 2:
            return 0.0
        soc = np.clip(soc, 0.0, 1.0)
        throughput = np.sum(np.abs(np.diff(soc)))
        return throughput / 2.0

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _generate_recommendation(self, metrics: Dict) -> str:
        """Generate recommendation based on comparison metrics"""
        improvements = sum([
            metrics['comfort']['violation_rate_diff'] < 0,
            metrics['cost']['net_cost_diff'] < 0,
            abs(metrics['flexibility']['battery_cycles_diff']) < 0.5,
            metrics['efficiency']['renewable_fraction_diff'] > 0,
            metrics['efficiency']['grid_dependency_diff'] < 0
        ])

        if improvements >= 4:
            return "Simulation 2 shows significant improvements across most metrics"
        elif improvements >= 3:
            return "Simulation 2 shows moderate improvements with some trade-offs"
        elif improvements >= 2:
            return "Mixed results - each simulation has different strengths"
        else:
            return "Simulation 1 performs better overall"

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    # You can customize the base path here
    base_path = os.environ.get("SIMULATION_BASE_PATH", "./simulation_results")

    server = BuildingSimulationMCPServer(base_path)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())