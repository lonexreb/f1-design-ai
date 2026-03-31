#!/usr/bin/env python3
"""
NVIDIA Omniverse Flow - F1 CFD Simulation Backend
==================================================
GPU-accelerated CFD simulation using NVIDIA Omniverse Flow as an alternative
to OpenFOAM. Provides the same SimulationResult interface for drop-in use
with the existing pipeline.

Requires: NVIDIA Omniverse Kit SDK with Flow extension installed.

Usage:
    # As module (called by run_pipeline.py)
    from omniverse_sim import find_omniverse, run_omniverse_cfd

    # Standalone test
    python3 scripts/omniverse_sim.py --test
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_to_usd import convert_stl_to_usd, DEFAULT_STL

# Omniverse constants
VELOCITY_MS = 83.33  # 300 km/h
AIR_DENSITY = 1.225  # kg/m³
REFERENCE_AREA = 1.5  # m² frontal area
CAR_LENGTH = 5.640   # m

# Common Omniverse install locations
OMNIVERSE_SEARCH_PATHS = [
    os.environ.get("OMNIVERSE_PATH", ""),
    os.path.expanduser("~/.local/share/ov/pkg"),
    os.path.expanduser("~/AppData/Local/ov/pkg"),
    "/opt/nvidia/omniverse",
    "/usr/local/omniverse",
]


@dataclass
class OmniverseConfig:
    """Configuration for Omniverse Flow simulation."""
    kit_path: Optional[str] = None
    flow_resolution: str = "medium"  # low, medium, high
    gpu_device: int = 0
    max_iterations: int = 2000
    convergence_threshold: float = 1e-5
    # Domain bounds (meters)
    domain_x: tuple = (-30.0, 60.0)
    domain_y: tuple = (-15.0, 15.0)
    domain_z: tuple = (0.0, 20.0)

    @property
    def resolution_cells(self) -> dict:
        """Map resolution name to approximate cell counts."""
        return {
            "low": {"base": [30, 10, 8], "max_cells": 500_000},
            "medium": {"base": [60, 20, 15], "max_cells": 5_000_000},
            "high": {"base": [120, 40, 30], "max_cells": 20_000_000},
        }[self.flow_resolution]


def find_omniverse() -> Optional[dict]:
    """
    Detect NVIDIA Omniverse installation.
    Returns dict with paths and capabilities, or None if not found.
    """
    result = {
        "kit_path": None,
        "has_flow": False,
        "has_physx": False,
        "has_warp": False,
        "version": None,
        "type": "omniverse",
    }

    # Check for Kit executable
    for search_path in OMNIVERSE_SEARCH_PATHS:
        if not search_path:
            continue
        path = Path(search_path)
        if not path.exists():
            continue

        # Look for kit executable
        kit_candidates = list(path.glob("**/kit"))
        if not kit_candidates:
            kit_candidates = list(path.glob("**/omni.kit*"))

        for kit in kit_candidates:
            if kit.is_file() and os.access(str(kit), os.X_OK):
                result["kit_path"] = str(kit)
                break

        if result["kit_path"]:
            break

    # Check for omni Python packages (available even without Kit)
    try:
        import omni.flow
        result["has_flow"] = True
    except ImportError:
        pass

    try:
        import omni.physx
        result["has_physx"] = True
    except ImportError:
        pass

    try:
        import warp as wp
        result["has_warp"] = True
        result["warp_version"] = wp.__version__
    except ImportError:
        pass

    # Check for NVIDIA GPU (required for any Omniverse simulation)
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if nvidia_smi.returncode == 0:
            result["gpu_info"] = nvidia_smi.stdout.strip()
        else:
            result["gpu_info"] = None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["gpu_info"] = None

    # Return None if nothing useful found
    if not result["kit_path"] and not result["has_flow"] and not result["has_warp"]:
        return None

    return result


def setup_flow_scene(usd_path: Path, config: OmniverseConfig) -> dict:
    """
    Configure Omniverse Flow wind tunnel scene.
    Returns scene configuration dict for the solver.
    """
    scene = {
        "usd_path": str(usd_path),
        "solver": "flow",
        "domain": {
            "x_range": config.domain_x,
            "y_range": config.domain_y,
            "z_range": config.domain_z,
        },
        "inlet": {
            "velocity": VELOCITY_MS,
            "direction": [1.0, 0.0, 0.0],
            "turbulence_intensity": 0.005,
        },
        "outlet": {
            "type": "pressure",
            "pressure": 101325.0,  # Pa (atmospheric)
        },
        "ground": {
            "type": "moving_wall",
            "velocity": VELOCITY_MS,
            "direction": [1.0, 0.0, 0.0],
        },
        "walls": {
            "top": "symmetry",
            "sides": "symmetry",
        },
        "fluid": {
            "density": AIR_DENSITY,
            "kinematic_viscosity": 1.516e-5,
        },
        "solver_settings": {
            "max_iterations": config.max_iterations,
            "convergence_threshold": config.convergence_threshold,
            "turbulence_model": "k_omega_sst",
            "gpu_device": config.gpu_device,
        },
        "mesh": config.resolution_cells,
        "force_monitoring": {
            "reference_area": REFERENCE_AREA,
            "reference_length": CAR_LENGTH,
            "reference_velocity": VELOCITY_MS,
            "reference_density": AIR_DENSITY,
            "report_interval": 10,
        },
    }
    return scene


def run_omniverse_flow(scene_config: dict) -> Optional[dict]:
    """
    Execute Omniverse Flow CFD simulation.
    Returns force coefficients dict or None on failure.

    NOTE: The omni.flow Python API below is a placeholder interface.
    NVIDIA Omniverse Flow configuration is currently done via USD prim
    attributes (omni.usd / pxr.Usd), not a high-level Solver class.
    This stub will be updated when the actual SDK integration is available.
    In practice, the ImportError path (_run_flow_via_kit) is taken.
    """
    try:
        import omni.flow as flow
    except ImportError:
        return _run_flow_via_kit(scene_config)

    # Placeholder API path (omni.flow high-level Solver is not yet public)
    solver = flow.Solver()
    solver.load_scene(scene_config["usd_path"])
    solver.set_domain(**scene_config["domain"])
    solver.set_inlet(**scene_config["inlet"])
    solver.set_outlet(**scene_config["outlet"])
    solver.set_ground(**scene_config["ground"])
    solver.set_fluid(**scene_config["fluid"])

    settings = scene_config["solver_settings"]
    solver.configure(
        max_iterations=settings["max_iterations"],
        convergence=settings["convergence_threshold"],
        turbulence_model=settings["turbulence_model"],
        device=settings["gpu_device"],
    )

    # Run simulation
    start_time = time.time()
    solver.initialize()
    converged = solver.solve()
    wall_time = time.time() - start_time

    # Extract forces
    forces = solver.get_force_coefficients(
        reference_area=scene_config["force_monitoring"]["reference_area"],
        reference_velocity=scene_config["force_monitoring"]["reference_velocity"],
        reference_density=scene_config["force_monitoring"]["reference_density"],
    )

    return {
        "cd": forces.drag_coefficient,
        "cl": forces.lift_coefficient,
        "cl_front": getattr(forces, "cl_front", 0.0),
        "cl_rear": getattr(forces, "cl_rear", 0.0),
        "converged": converged,
        "iterations": solver.iteration_count,
        "wall_time_s": wall_time,
    }


def _run_flow_via_kit(scene_config: dict) -> Optional[dict]:
    """
    Run Flow simulation via Omniverse Kit CLI as subprocess.
    Fallback when omni.flow Python module isn't directly importable.
    """
    kit_path = None
    for search_path in OMNIVERSE_SEARCH_PATHS:
        if not search_path:
            continue
        path = Path(search_path)
        kits = list(path.glob("**/kit"))
        if kits:
            kit_path = str(kits[0])
            break

    if not kit_path:
        print("  WARNING: Omniverse Kit not found for Flow simulation")
        return None

    # Write scene config to secure temp file for Kit to read
    config_fd, config_path_str = tempfile.mkstemp(suffix=".json", prefix="f1_flow_config_")
    config_path = Path(config_path_str)
    results_fd, results_path_str = tempfile.mkstemp(suffix=".json", prefix="f1_flow_results_")
    results_path = Path(results_path_str)
    os.close(results_fd)

    try:
        with os.fdopen(config_fd, "w") as f:
            json.dump(scene_config, f, indent=2)

        # Kit CLI execution with Flow extension
        cmd = [
            kit_path,
            "--enable", "omni.flow",
            "--exec", f"flow.run_simulation('{config_path}', '{results_path}')",
            "--no-window",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=7200, cwd=str(PROJECT_DIR)
        )
        if result.returncode == 0 and results_path.exists():
            return json.loads(results_path.read_text())
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  WARNING: Kit execution failed: {e}")
    finally:
        config_path.unlink(missing_ok=True)
        results_path.unlink(missing_ok=True)

    return None


def run_warp_cfd(usd_path: Path, config: OmniverseConfig) -> Optional[dict]:
    """
    Lightweight GPU CFD using NVIDIA Warp.
    Warp provides GPU-accelerated custom physics kernels without
    the full Omniverse Kit dependency. Useful for rapid prototyping.

    NOTE: LBM kernel not yet implemented — returns None to trigger fallback.
    """
    try:
        import warp as wp
    except ImportError:
        print("  WARNING: NVIDIA Warp not available")
        return None

    res = config.resolution_cells
    nx, ny, nz = res["base"][0] * 4, res["base"][1] * 4, res["base"][2] * 4

    print(f"  Warp LBM solver not yet implemented ({nx}x{ny}x{nz} grid planned)")
    print("  NOTE: Use Omniverse Flow for production GPU CFD")

    return None  # Triggers fallback


def run_omniverse_cfd(params, config: Optional[OmniverseConfig] = None) -> Optional[dict]:
    """
    Main entry point: run Omniverse-based CFD for given SimulationParams.
    Tries Flow first, then Warp, returns None if both unavailable.

    Args:
        params: SimulationParams dataclass from run_pipeline.py
        config: Optional OmniverseConfig (uses defaults if None)

    Returns:
        dict with cd, cl, ld_ratio, converged, iterations, wall_time_s
        or None if Omniverse is not available
    """
    if config is None:
        config = OmniverseConfig()

    print(f"\n{'='*60}")
    print("  NVIDIA Omniverse CFD Simulation")
    print(f"{'='*60}")

    # Step 1: Generate geometry for the given params via Blender, then convert to USD
    stl_path = DEFAULT_STL
    usd_path = PROJECT_DIR / "omniverse" / "f1_car.usda"

    # Re-generate STL if params differ from default (trigger Blender re-run)
    if hasattr(params, "ride_height"):
        try:
            blender_script = PROJECT_DIR / "blender" / "f1_car_generator.py"
            stl_dir = stl_path.parent
            cmd = [
                "blender", "--background", "--python", str(blender_script),
                "--",
                "--ride-height", str(params.ride_height),
                "--front-wing-angle", str(params.front_wing_angle),
                "--rear-wing-angle", str(params.rear_wing_angle),
                "--diffuser-angle", str(params.diffuser_angle),
                "--sidepod-undercut", str(params.sidepod_undercut),
                "--output-dir", str(stl_dir),
            ]
            subprocess.run(cmd, capture_output=True, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Blender not available; use existing STL if present

    # Convert STL -> USD if needed (stale or missing)
    needs_conversion = (
        not usd_path.exists() or
        (stl_path.exists() and stl_path.stat().st_mtime > usd_path.stat().st_mtime)
    )
    if needs_conversion:
        print("  Converting STL → USD...")
        if not convert_stl_to_usd(stl_path, usd_path, with_wind_tunnel=True):
            print("  ERROR: USD conversion failed")
            return None

    # Step 2: Setup scene
    scene_config = setup_flow_scene(usd_path, config)
    print(f"  Domain: {config.domain_x[0]}m to {config.domain_x[1]}m (streamwise)")
    print(f"  Inlet: {VELOCITY_MS} m/s ({VELOCITY_MS * 3.6:.0f} km/h)")
    print(f"  Resolution: {config.flow_resolution} ({config.resolution_cells['max_cells']:,} cells max)")

    # Step 3: Try Flow, then Warp
    ov_info = find_omniverse()
    result = None

    if ov_info and (ov_info.get("has_flow") or ov_info.get("kit_path")):
        print("  Backend: Omniverse Flow (GPU RANS)")
        result = run_omniverse_flow(scene_config)

    if result is None and ov_info and ov_info.get("has_warp"):
        print("  Backend: NVIDIA Warp (GPU LBM, experimental)")
        result = run_warp_cfd(usd_path, config)

    if result is None:
        print("  WARNING: No Omniverse backends available")
        return None

    # Compute L/D
    if result.get("cd", 0) != 0:
        result["ld_ratio"] = round(abs(result.get("cl", 0)) / result["cd"], 2)
    else:
        result["ld_ratio"] = 0.0

    result["notes"] = f"Omniverse Flow (GPU, {config.flow_resolution} resolution)"

    print("\n  Results:")
    print(f"    Cd = {result.get('cd', 'N/A')}")
    print(f"    Cl = {result.get('cl', 'N/A')}")
    print(f"    L/D = {result.get('ld_ratio', 'N/A')}")
    print(f"    Converged: {result.get('converged', False)}")
    print(f"    Wall time: {result.get('wall_time_s', 0):.1f}s")

    return result


def main():
    """Standalone test / status check."""
    print("=" * 60)
    print("  NVIDIA Omniverse - F1 CFD Backend Status")
    print("=" * 60)

    ov = find_omniverse()
    if ov is None:
        print("\n  Status: NOT AVAILABLE")
        print("\n  To install NVIDIA Omniverse:")
        print("  1. Download Omniverse Launcher: https://www.nvidia.com/omniverse")
        print("  2. Install Kit SDK with Flow extension")
        print("  3. Or: pip install nvidia-warp (lightweight alternative)")
        print("\n  Pipeline will fall back to OpenFOAM or empirical estimates.")
    else:
        print(f"\n  Kit path:   {ov.get('kit_path', 'Not found')}")
        print(f"  Flow:       {'Available' if ov.get('has_flow') else 'Not found'}")
        print(f"  PhysX:      {'Available' if ov.get('has_physx') else 'Not found'}")
        print(f"  Warp:       {'Available' if ov.get('has_warp') else 'Not found'}")
        print(f"  GPU:        {ov.get('gpu_info', 'Not detected')}")

    # Test USD conversion if STL exists
    if DEFAULT_STL.exists():
        print(f"\n  STL found: {DEFAULT_STL}")
        print("  Run with --test to attempt full simulation")
    else:
        print("\n  No STL found. Generate first with:")
        print("  blender --background --python blender/f1_car_generator.py")

    if "--test" in sys.argv:
        from run_pipeline import SimulationParams
        params = SimulationParams()
        result = run_omniverse_cfd(params)
        if result:
            print(f"\n  Test result: {json.dumps(result, indent=2)}")
        else:
            print("\n  Test: Omniverse simulation not available, would fall back to OpenFOAM")


if __name__ == "__main__":
    main()
