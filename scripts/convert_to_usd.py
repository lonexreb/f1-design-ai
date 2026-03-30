#!/usr/bin/env python3
"""
STL to USD Converter for NVIDIA Omniverse
==========================================
Converts Blender-generated F1 car STL geometry to Universal Scene Description (USD)
format for use with NVIDIA Omniverse Flow simulation and visualization.

Usage:
    python3 scripts/convert_to_usd.py
    python3 scripts/convert_to_usd.py --input path/to/car.stl --output omniverse/car.usda
    python3 scripts/convert_to_usd.py --with-wind-tunnel
"""

import argparse
import struct
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_STL = PROJECT_DIR / "openfoam" / "f1_baseline" / "constant" / "triSurface" / "f1_car_combined.stl"
DEFAULT_USD = PROJECT_DIR / "omniverse" / "f1_car.usda"

# FIA 2024 reference constants
VELOCITY_MS = 83.33  # 300 km/h
DOMAIN_BOUNDS = {
    "x_min": -30.0, "x_max": 60.0,
    "y_min": -15.0, "y_max": 15.0,
    "z_min": 0.0,   "z_max": 20.0,
}


def read_stl_binary(stl_path: Path) -> tuple:
    """Read binary STL file, return (vertices, normals, faces)."""
    with open(stl_path, "rb") as f:
        header = f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]

        vertices = []
        normals = []
        face_indices = []

        for i in range(num_triangles):
            nx, ny, nz = struct.unpack("<3f", f.read(12))
            normals.append((nx, ny, nz))

            tri_verts = []
            for _ in range(3):
                vx, vy, vz = struct.unpack("<3f", f.read(12))
                tri_verts.append((vx, vy, vz))
                vertices.append((vx, vy, vz))

            base = i * 3
            face_indices.append((base, base + 1, base + 2))

            f.read(2)  # attribute byte count

    return vertices, normals, face_indices, num_triangles


def read_stl_ascii(stl_path: Path) -> tuple:
    """Read ASCII STL file, return (vertices, normals, faces)."""
    vertices = []
    normals = []
    face_indices = []
    tri_count = 0

    with open(stl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                normals.append((float(parts[2]), float(parts[3]), float(parts[4])))
            elif line.startswith("vertex"):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(vertices) % 3 == 0:
                    base = tri_count * 3
                    face_indices.append((base, base + 1, base + 2))
                    tri_count += 1

    return vertices, normals, face_indices, tri_count


def read_stl(stl_path: Path) -> tuple:
    """Auto-detect STL format and read."""
    with open(stl_path, "rb") as f:
        header = f.read(80)

    if b"solid" in header[:10]:
        # Could be ASCII, verify
        with open(stl_path, "r", errors="ignore") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            if second_line.startswith("facet") or second_line.startswith("endsolid"):
                return read_stl_ascii(stl_path)

    return read_stl_binary(stl_path)


def generate_usda(vertices, normals, face_indices, num_triangles,
                  with_wind_tunnel: bool = False) -> str:
    """Generate USD ASCII (.usda) content from mesh data."""

    # Format vertices as USD array
    points_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in vertices)

    # Face vertex counts (all triangles = 3)
    face_counts_str = ", ".join(["3"] * num_triangles)

    # Face vertex indices (flattened)
    face_indices_flat = []
    for face in face_indices:
        face_indices_flat.extend(face)
    face_indices_str = ", ".join(str(i) for i in face_indices_flat)

    # Normals per-face
    normals_str = ", ".join(f"({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f})" for n in normals)

    usda = f'''#usda 1.0
(
    defaultPrim = "F1_WindTunnel"
    metersPerUnit = 1.0
    upAxis = "Z"
    doc = "F1 Car Aerodynamic Simulation Scene - Generated for NVIDIA Omniverse"
)

def Xform "F1_WindTunnel" (
    kind = "assembly"
)
{{
    # F1 Car Geometry - imported from Blender parametric generator
    def Mesh "F1_Car" (
        kind = "component"
    )
    {{
        # Geometry data ({num_triangles} triangles, {len(vertices)} vertices)
        int[] faceVertexCounts = [{face_counts_str}]
        int[] faceVertexIndices = [{face_indices_str}]
        point3f[] points = [{points_str}]
        normal3f[] normals = [{normals_str}]
        uniform token subdivisionScheme = "none"

        # Surface material - carbon fiber composite
        rel material:binding = </F1_WindTunnel/Materials/CarbonFiber>

        # Physics properties for Omniverse Flow
        bool physics:rigidBodyEnabled = 1
        bool physics:collisionEnabled = 1
    }}
'''

    if with_wind_tunnel:
        bounds = DOMAIN_BOUNDS
        usda += f'''
    # Wind tunnel domain boundaries
    def Xform "WindTunnel"
    {{
        def Cube "Domain" {{
            float3 xformOp:translate = ({(bounds["x_min"] + bounds["x_max"]) / 2:.1f}, {(bounds["y_min"] + bounds["y_max"]) / 2:.1f}, {(bounds["z_min"] + bounds["z_max"]) / 2:.1f})
            float3 xformOp:scale = ({(bounds["x_max"] - bounds["x_min"]) / 2:.1f}, {(bounds["y_max"] - bounds["y_min"]) / 2:.1f}, {(bounds["z_max"] - bounds["z_min"]) / 2:.1f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            bool visibility = 0
        }}

        # Inlet boundary (upstream face)
        def Xform "Inlet" {{
            float3 xformOp:translate = ({bounds["x_min"]:.1f}, 0.0, {bounds["z_max"] / 2:.1f})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            # Flow velocity: {VELOCITY_MS} m/s (300 km/h)
            custom float flow:velocity = {VELOCITY_MS}
            custom float3 flow:direction = (1.0, 0.0, 0.0)
            custom float flow:turbulenceIntensity = 0.005
        }}

        # Ground plane (moving wall)
        def Plane "Ground" {{
            float3 xformOp:translate = (15.0, 0.0, 0.0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            custom float flow:wallVelocity = {VELOCITY_MS}
            custom float3 flow:wallDirection = (1.0, 0.0, 0.0)
        }}

        # Outlet boundary (downstream face)
        def Xform "Outlet" {{
            float3 xformOp:translate = ({bounds["x_max"]:.1f}, 0.0, {bounds["z_max"] / 2:.1f})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            custom token flow:boundaryType = "pressure_outlet"
            custom float flow:pressure = 101325.0
        }}
    }}
'''

    # Materials
    usda += '''
    # Materials
    def Scope "Materials"
    {
        def Material "CarbonFiber"
        {
            token outputs:surface.connect = </F1_WindTunnel/Materials/CarbonFiber/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.05, 0.05, 0.08)
                float inputs:roughness = 0.15
                float inputs:metallic = 0.3
                float inputs:specularLevel = 0.8
                token outputs:surface
            }
        }
    }
}
'''
    return usda


def convert_stl_to_usd(stl_path: Path, usd_path: Path,
                        with_wind_tunnel: bool = False) -> bool:
    """Convert STL file to USD format."""
    print(f"  Reading STL: {stl_path}")
    if not stl_path.exists():
        print(f"  ERROR: STL file not found: {stl_path}")
        print("  Run Blender first: blender --background --python blender/f1_car_generator.py")
        return False

    vertices, normals, face_indices, num_triangles = read_stl(stl_path)
    print(f"  Loaded: {num_triangles:,} triangles, {len(vertices):,} vertices")

    print(f"  Generating USD{'A' if str(usd_path).endswith('.usda') else 'C'}...")
    usda_content = generate_usda(vertices, normals, face_indices, num_triangles,
                                  with_wind_tunnel=with_wind_tunnel)

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text(usda_content)

    size_mb = usd_path.stat().st_size / (1024 * 1024)
    print(f"  Written: {usd_path} ({size_mb:.1f} MB)")

    if with_wind_tunnel:
        print(f"  Wind tunnel configured: {VELOCITY_MS} m/s inlet, moving ground")

    return True


def convert_with_pxr(stl_path: Path, usd_path: Path,
                     with_wind_tunnel: bool = False) -> bool:
    """Convert using official OpenUSD Python library (pxr) if available."""
    try:
        from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade
    except ImportError:
        print("  pxr (OpenUSD) not installed, using built-in USDA writer")
        return False

    vertices, normals, face_indices, num_triangles = read_stl(stl_path)
    print(f"  Loaded: {num_triangles:,} triangles, {len(vertices):,} vertices")
    print(f"  Using pxr (OpenUSD) for USD generation...")

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(stage.DefinePrim("/F1_WindTunnel", "Xform"))

    # Create car mesh
    mesh = UsdGeom.Mesh.Define(stage, "/F1_WindTunnel/F1_Car")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * num_triangles)
    face_flat = []
    for f in face_indices:
        face_flat.extend(f)
    mesh.CreateFaceVertexIndicesAttr(face_flat)
    mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
    mesh.SetNormalsInterpolation("uniform")
    mesh.CreateSubdivisionSchemeAttr("none")

    stage.GetRootLayer().Save()
    size_mb = usd_path.stat().st_size / (1024 * 1024)
    print(f"  Written: {usd_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert F1 car STL to USD for Omniverse")
    parser.add_argument("--input", type=Path, default=DEFAULT_STL,
                        help=f"Input STL path (default: {DEFAULT_STL.relative_to(PROJECT_DIR)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_USD,
                        help=f"Output USD path (default: {DEFAULT_USD.relative_to(PROJECT_DIR)})")
    parser.add_argument("--with-wind-tunnel", action="store_true",
                        help="Include wind tunnel domain, inlet/outlet/ground in USD scene")
    parser.add_argument("--force-builtin", action="store_true",
                        help="Use built-in USDA writer even if pxr is available")
    args = parser.parse_args()

    print("=" * 60)
    print("  STL → USD Converter for NVIDIA Omniverse")
    print("=" * 60)

    success = False
    if not args.force_builtin:
        success = convert_with_pxr(args.input, args.output, args.with_wind_tunnel)

    if not success:
        success = convert_stl_to_usd(args.input, args.output, args.with_wind_tunnel)

    if success:
        print("\n  USD file ready for NVIDIA Omniverse.")
        print("  Open in Omniverse Create/Code or load via Kit API.")
    else:
        print("\n  Conversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
