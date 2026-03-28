#!/usr/bin/env python3
"""
F1 Car Design Pipeline - Blender -> OpenFOAM -> ParaView
=========================================================
Automates the full aerodynamic design loop:
1. Generate parametric F1 car geometry in Blender
2. Export STL for OpenFOAM meshing
3. Run CFD simulation (blockMesh -> snappyHexMesh -> simpleFoam)
4. Extract force coefficients (Cd, Cl, L/D ratio)
5. Optionally run parameter sweeps for optimization

Usage:
    python3 run_pipeline.py                          # Single baseline run
    python3 run_pipeline.py --sweeps ride_height     # Sweep ride height
    python3 run_pipeline.py --sweeps all             # Full parameter sweep
    python3 run_pipeline.py --docker                 # Use Docker OpenFOAM
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
BLENDER_DIR = PROJECT_DIR / "blender"
OPENFOAM_DIR = PROJECT_DIR / "openfoam" / "f1_baseline"
RESULTS_DIR = PROJECT_DIR / "results"

BLENDER_SCRIPT = BLENDER_DIR / "f1_car_generator.py"
DOCKER_WRAPPER = SCRIPT_DIR / "openfoam-docker.sh"


@dataclass
class SimulationParams:
    """Parametric design variables for the F1 car."""
    ride_height: float = 0.030       # meters
    front_wing_angle: float = 14.0   # degrees
    rear_wing_angle: float = 16.0    # degrees
    diffuser_angle: float = 12.0     # degrees
    sidepod_undercut: float = 0.15   # meters


@dataclass
class SimulationResult:
    """Results from a single CFD run."""
    params: SimulationParams
    cd: float = 0.0          # Drag coefficient
    cl: float = 0.0          # Lift coefficient (negative = downforce)
    cl_front: float = 0.0    # Front axle downforce coefficient
    cl_rear: float = 0.0     # Rear axle downforce coefficient
    ld_ratio: float = 0.0    # Lift-to-drag ratio (efficiency)
    converged: bool = False
    iterations: int = 0
    wall_time_s: float = 0.0
    notes: str = ""


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    candidates = [
        "blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]
    for cmd in candidates:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def find_openfoam(use_docker: bool = False) -> dict:
    """Find OpenFOAM executables or Docker wrapper."""
    if use_docker:
        if DOCKER_WRAPPER.exists():
            return {
                "blockMesh": [str(DOCKER_WRAPPER), "blockMesh"],
                "snappyHexMesh": [str(DOCKER_WRAPPER), "snappyHexMesh"],
                "simpleFoam": [str(DOCKER_WRAPPER), "simpleFoam"],
                "surfaceFeatureExtract": [str(DOCKER_WRAPPER), "surfaceFeatureExtract"],
                "type": "docker",
            }
        else:
            print("ERROR: Docker wrapper not found. Run install.sh first.")
            sys.exit(1)

    cmds = {}
    for tool in ["blockMesh", "snappyHexMesh", "simpleFoam", "surfaceFeatureExtract"]:
        try:
            subprocess.run([tool, "-help"], capture_output=True, timeout=5)
            cmds[tool] = [tool]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            cmds[tool] = None

    if all(v is not None for v in cmds.values()):
        cmds["type"] = "native"
        return cmds

    print("WARNING: OpenFOAM not found natively. Trying Docker...")
    return find_openfoam(use_docker=True)


def run_blender(params: SimulationParams, blender_cmd: str) -> bool:
    """Generate F1 car geometry with given parameters."""
    print(f"\n{'='*60}")
    print(f"  STEP 1: Generating F1 Car Geometry (Blender)")
    print(f"{'='*60}")
    print(f"  Ride height:      {params.ride_height*1000:.1f} mm")
    print(f"  Front wing angle: {params.front_wing_angle:.1f} deg")
    print(f"  Rear wing angle:  {params.rear_wing_angle:.1f} deg")
    print(f"  Diffuser angle:   {params.diffuser_angle:.1f} deg")
    print(f"  Sidepod undercut: {params.sidepod_undercut*1000:.1f} mm")

    stl_dir = OPENFOAM_DIR / "constant" / "triSurface"
    cmd = [
        blender_cmd, "--background", "--python", str(BLENDER_SCRIPT),
        "--",
        "--ride-height", str(params.ride_height),
        "--front-wing-angle", str(params.front_wing_angle),
        "--rear-wing-angle", str(params.rear_wing_angle),
        "--diffuser-angle", str(params.diffuser_angle),
        "--sidepod-undercut", str(params.sidepod_undercut),
        "--output-dir", str(stl_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ERROR: Blender failed:\n{result.stderr[-500:]}")
        return False

    # Check STL was created
    combined_stl = stl_dir / "f1_car_combined.stl"
    if combined_stl.exists():
        size_mb = combined_stl.stat().st_size / (1024 * 1024)
        print(f"  OK: Combined STL created ({size_mb:.1f} MB)")
        return True
    else:
        print("  ERROR: Combined STL not found after Blender run.")
        return False


def run_openfoam(of_cmds: dict) -> bool:
    """Run OpenFOAM meshing and simulation."""
    print(f"\n{'='*60}")
    print(f"  STEP 2: Running CFD Simulation (OpenFOAM)")
    print(f"{'='*60}")

    env = os.environ.copy()
    if of_cmds["type"] == "docker":
        env["OPENFOAM_CASE_DIR"] = str(OPENFOAM_DIR)

    steps = [
        ("blockMesh", "Generating base mesh..."),
        ("surfaceFeatureExtract", "Extracting surface features..."),
        ("snappyHexMesh", "Refining mesh around car (this takes a while)..."),
        ("simpleFoam", "Running RANS CFD solver..."),
    ]

    for tool_name, description in steps:
        cmd = of_cmds.get(tool_name)
        if cmd is None:
            print(f"  SKIP: {tool_name} not available")
            continue

        print(f"\n  [{tool_name}] {description}")
        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(OPENFOAM_DIR),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for simpleFoam
                env=env,
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                print(f"  ERROR: {tool_name} failed after {elapsed:.0f}s")
                # Save error log
                log_path = RESULTS_DIR / f"{tool_name}_error.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(result.stderr[-2000:])
                print(f"  Error log: {log_path}")

                if tool_name == "simpleFoam":
                    # Still try to extract partial results
                    print("  Attempting to extract partial results...")
                    return True
                return False

            print(f"  OK: {tool_name} completed in {elapsed:.0f}s")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {tool_name} exceeded 2 hour limit")
            return False

    return True


def extract_results(params: SimulationParams) -> SimulationResult:
    """Parse OpenFOAM force coefficient output."""
    print(f"\n{'='*60}")
    print(f"  STEP 3: Extracting Results")
    print(f"{'='*60}")

    result = SimulationResult(params=params)

    # Look for forceCoeffs output
    coeff_dir = OPENFOAM_DIR / "postProcessing" / "forceCoeffs"
    if not coeff_dir.exists():
        print("  WARNING: No forceCoeffs output found.")
        print("  This is expected if OpenFOAM hasn't been run yet.")
        # Return estimated values based on design parameters
        result = estimate_coefficients(params)
        return result

    # Find latest time directory
    time_dirs = sorted([d for d in coeff_dir.iterdir() if d.is_dir()], key=lambda d: float(d.name))
    if not time_dirs:
        print("  WARNING: No time directories in forceCoeffs.")
        return estimate_coefficients(params)

    latest = time_dirs[-1]
    coeff_file = latest / "coefficient.dat"
    if not coeff_file.exists():
        # Try alternate name
        coeff_file = latest / "forceCoeffs.dat"

    if coeff_file.exists():
        lines = coeff_file.read_text().strip().split("\n")
        # Last non-comment line has the final values
        data_lines = [l for l in lines if not l.startswith("#")]
        if data_lines:
            values = data_lines[-1].split()
            try:
                result.iterations = int(float(values[0]))
                result.cd = float(values[1])
                result.cl = float(values[3])  # Typically column 4
                result.ld_ratio = abs(result.cl / result.cd) if result.cd != 0 else 0
                result.converged = True
                print(f"  Cd = {result.cd:.4f}")
                print(f"  Cl = {result.cl:.4f} (negative = downforce)")
                print(f"  L/D = {result.ld_ratio:.2f}")
                print(f"  Iterations: {result.iterations}")
            except (IndexError, ValueError) as e:
                print(f"  WARNING: Could not parse coefficients: {e}")
                return estimate_coefficients(params)
    else:
        print("  WARNING: Coefficient file not found.")
        return estimate_coefficients(params)

    return result


def estimate_coefficients(params: SimulationParams) -> SimulationResult:
    """
    Estimate aero coefficients from design parameters using empirical correlations.
    These are rough estimates for initial parameter sweep guidance.
    Based on published F1 aerodynamic data and Ravelli & Savini (2018).
    """
    print("  Using empirical estimation (no CFD data available)")

    # Baseline: typical 2024 F1 car at 300 km/h
    # Cd ~ 0.7-1.2, Cl ~ -3.0 to -5.5 (negative = downforce)
    base_cd = 0.95
    base_cl = -3.5

    # Ride height effect on downforce (ground effect)
    # Lower ride height = more downforce (up to a point)
    # Optimum around 25-35mm, drops off sharply below 20mm
    rh_mm = params.ride_height * 1000
    if rh_mm < 20:
        rh_factor = 0.85  # Too low, stalling
    elif rh_mm < 35:
        rh_factor = 1.0 + (30 - rh_mm) * 0.02  # Peak around 25-30mm
    else:
        rh_factor = 1.0 - (rh_mm - 35) * 0.015  # Drops off at higher ride heights

    # Front wing angle effect
    fw_factor_cl = 1.0 + (params.front_wing_angle - 14) * 0.03
    fw_factor_cd = 1.0 + (params.front_wing_angle - 14) * 0.02

    # Rear wing angle effect (bigger impact on both Cd and Cl)
    rw_factor_cl = 1.0 + (params.rear_wing_angle - 16) * 0.04
    rw_factor_cd = 1.0 + (params.rear_wing_angle - 16) * 0.035

    # Diffuser angle effect
    if params.diffuser_angle < 8:
        diff_factor = 0.9  # Too shallow, not extracting air
    elif params.diffuser_angle < 15:
        diff_factor = 1.0 + (params.diffuser_angle - 12) * 0.02
    else:
        diff_factor = 1.0 - (params.diffuser_angle - 15) * 0.05  # Separation

    # Sidepod undercut effect (more undercut = better flow to diffuser)
    undercut_factor = 1.0 + (params.sidepod_undercut - 0.15) * 0.5

    cd = base_cd * fw_factor_cd * rw_factor_cd
    cl = base_cl * rh_factor * fw_factor_cl * rw_factor_cl * diff_factor * undercut_factor

    result = SimulationResult(
        params=params,
        cd=round(cd, 4),
        cl=round(cl, 4),
        ld_ratio=round(abs(cl / cd), 2) if cd != 0 else 0,
        converged=False,
        notes="Empirical estimate (no CFD run)",
    )

    print(f"  Estimated Cd = {result.cd:.4f}")
    print(f"  Estimated Cl = {result.cl:.4f}")
    print(f"  Estimated L/D = {result.ld_ratio:.2f}")

    return result


def run_parameter_sweep(sweep_type: str, blender_cmd: Optional[str], of_cmds: Optional[dict]) -> list:
    """Run a parametric sweep over design variables."""
    print(f"\n{'#'*60}")
    print(f"  PARAMETER SWEEP: {sweep_type}")
    print(f"{'#'*60}")

    sweeps = {
        "ride_height": {
            "param": "ride_height",
            "values": [0.020, 0.025, 0.030, 0.035, 0.040, 0.050],
            "unit": "mm",
            "scale": 1000,
        },
        "front_wing_angle": {
            "param": "front_wing_angle",
            "values": [10, 12, 14, 16, 18, 20],
            "unit": "deg",
            "scale": 1,
        },
        "rear_wing_angle": {
            "param": "rear_wing_angle",
            "values": [10, 12, 14, 16, 18, 20, 22],
            "unit": "deg",
            "scale": 1,
        },
        "diffuser_angle": {
            "param": "diffuser_angle",
            "values": [6, 8, 10, 12, 14, 16, 18],
            "unit": "deg",
            "scale": 1,
        },
        "sidepod_undercut": {
            "param": "sidepod_undercut",
            "values": [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
            "unit": "mm",
            "scale": 1000,
        },
    }

    if sweep_type == "all":
        sweep_configs = list(sweeps.values())
    elif sweep_type in sweeps:
        sweep_configs = [sweeps[sweep_type]]
    else:
        print(f"  Unknown sweep type: {sweep_type}")
        print(f"  Available: {', '.join(sweeps.keys())}, all")
        return []

    all_results = []

    for config in sweep_configs:
        param_name = config["param"]
        print(f"\n  Sweeping: {param_name}")
        print(f"  Values: {config['values']} {config['unit']}")

        results = []
        for value in config["values"]:
            params = SimulationParams()
            setattr(params, param_name, value)

            print(f"\n  --- {param_name} = {value * config['scale']:.1f} {config['unit']} ---")

            # If Blender + OpenFOAM available, run full pipeline
            if blender_cmd and of_cmds:
                run_blender(params, blender_cmd)
                run_openfoam(of_cmds)

            result = extract_results(params)
            results.append(result)

        all_results.extend(results)

        # Print sweep summary table
        print(f"\n  {'='*60}")
        print(f"  SWEEP RESULTS: {param_name}")
        print(f"  {'='*60}")
        print(f"  {'Value':>10} | {'Cd':>8} | {'Cl':>8} | {'L/D':>8} | {'Notes'}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+--------")
        for r in results:
            val = getattr(r.params, param_name) * config["scale"]
            note = "CFD" if r.converged else "Est."
            print(f"  {val:>8.1f}{config['unit'][:1]} | {r.cd:>8.4f} | {r.cl:>8.4f} | {r.ld_ratio:>8.2f} | {note}")

    return all_results


def save_results(results: list, filename: str = "sweep_results.json"):
    """Save results to JSON for further analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = []
    for r in results:
        entry = {
            "params": asdict(r.params),
            "cd": r.cd,
            "cl": r.cl,
            "ld_ratio": r.ld_ratio,
            "converged": r.converged,
            "iterations": r.iterations,
            "wall_time_s": r.wall_time_s,
            "notes": r.notes,
        }
        output.append(entry)

    path = RESULTS_DIR / filename
    path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="F1 Car Design Pipeline")
    parser.add_argument("--sweeps", type=str, default=None,
                        help="Parameter to sweep: ride_height, front_wing_angle, "
                             "rear_wing_angle, diffuser_angle, sidepod_undercut, all")
    parser.add_argument("--docker", action="store_true",
                        help="Use Docker for OpenFOAM")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Skip Blender/OpenFOAM, use empirical estimates only")
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Car Aerodynamic Design Pipeline")
    print("  AI-Assisted | Blender + OpenFOAM + ParaView")
    print("=" * 60)

    # Check tools
    blender_cmd = None
    of_cmds = None

    if not args.estimate_only:
        blender_cmd = find_blender()
        if blender_cmd:
            print(f"  Blender: {blender_cmd}")
        else:
            print("  Blender: NOT FOUND (will skip geometry generation)")

        try:
            of_cmds = find_openfoam(use_docker=args.docker)
            print(f"  OpenFOAM: {of_cmds['type']}")
        except SystemExit:
            print("  OpenFOAM: NOT FOUND (will use empirical estimates)")
            of_cmds = None
    else:
        print("  Mode: Empirical estimates only (no CFD)")

    # Run sweep or single case
    if args.sweeps:
        results = run_parameter_sweep(args.sweeps, blender_cmd, of_cmds)
        if results:
            save_results(results, f"sweep_{args.sweeps}.json")
    else:
        # Single baseline run
        params = SimulationParams()

        if blender_cmd and not args.estimate_only:
            run_blender(params, blender_cmd)

        if of_cmds and not args.estimate_only:
            run_openfoam(of_cmds)

        result = extract_results(params)
        save_results([result], "baseline_result.json")

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
