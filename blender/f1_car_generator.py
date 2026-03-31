"""
F1 Car Parametric Geometry Generator for Blender
=================================================
Generates a simplified but aerodynamically representative F1 car geometry
based on Red Bull RB19/RB20 design philosophy.

Usage:
    blender --background --python f1_car_generator.py
    blender --background --python f1_car_generator.py -- --ride-height 0.03 --rear-wing-angle 18

Key aerodynamic surfaces modeled:
    - Monocoque/chassis body
    - Front wing (multi-element)
    - Rear wing (DRS-capable profile)
    - Underbody floor with Venturi tunnels
    - Diffuser
    - Sidepods (RB20-style shark inlet + undercut)
    - Halo
    - Wheels (simplified)

All dimensions in meters, based on 2024 FIA Technical Regulations.
"""

import sys
import math
import argparse
import os

# Handle Blender's argument parsing (-- separates Blender args from script args)
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser(description="F1 Car Parametric Generator")
parser.add_argument("--ride-height", type=float, default=0.030,
                    help="Ground clearance in meters (default: 0.030)")
parser.add_argument("--front-wing-angle", type=float, default=14.0,
                    help="Front wing angle of attack in degrees (default: 14)")
parser.add_argument("--rear-wing-angle", type=float, default=16.0,
                    help="Rear wing angle of attack in degrees (default: 16)")
parser.add_argument("--diffuser-angle", type=float, default=12.0,
                    help="Diffuser exit angle in degrees (default: 12)")
parser.add_argument("--sidepod-undercut", type=float, default=0.15,
                    help="Sidepod undercut depth in meters (default: 0.15)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for STL files")
parser.add_argument("--output-blend", type=str, default=None,
                    help="Output .blend file path")
params = parser.parse_args(argv)

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    print("WARNING: Running outside Blender. Generating config only.")

# =============================================================================
# FIA 2024 Technical Regulation Reference Dimensions
# =============================================================================
CAR_LENGTH = 5.640           # Max overall length
CAR_WIDTH = 2.000            # Max overall width
CAR_HEIGHT = 0.950           # Max height (excl. roll structure)
WHEELBASE = 3.600            # Typical wheelbase
TRACK_FRONT = 1.600          # Front track width
TRACK_REAR = 1.580           # Rear track width
WHEEL_DIAMETER = 0.720       # 18-inch wheel diameter
WHEEL_WIDTH_FRONT = 0.305    # Front tire width
WHEEL_WIDTH_REAR = 0.405     # Rear tire width
REFERENCE_AREA = 1.5         # Frontal reference area (m^2) for Cd calculation
FLOOR_WIDTH = 1.600          # Underbody reference plane width

# Derived
HALF_WIDTH = CAR_WIDTH / 2
HALF_FLOOR = FLOOR_WIDTH / 2


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def create_material(name, color):
    """Create a simple material with given RGBA color."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat


def assign_material(obj, mat):
    """Assign material to object."""
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# =============================================================================
# Geometry Generators
# =============================================================================

def create_monocoque():
    """
    Create the main chassis/monocoque body.
    RB-style: narrow nose, V-shaped chassis opening volume for underfloor.
    """
    verts = []
    faces = []

    # Cross-section profiles at various X stations (front to rear)
    # Each profile: (x, [(y, z), ...]) defining half-section (mirrored)
    nose_tip_x = -2.5         # Nose tip
    nose_end_x = -1.5         # Where nose widens to cockpit
    cockpit_front_x = -0.8    # Front of cockpit opening
    cockpit_rear_x = 0.2      # Rear of cockpit opening
    engine_cover_x = 1.0      # Engine cover start
    rear_x = 2.8              # Rear of bodywork

    ride_h = params.ride_height

    profiles = [
        # (x_position, width_at_top, height_top, width_at_bottom, height_bottom)
        (nose_tip_x,       0.05,  ride_h + 0.20, 0.05,  ride_h + 0.05),
        (nose_tip_x + 0.3, 0.12,  ride_h + 0.28, 0.10,  ride_h + 0.05),
        (nose_end_x,       0.30,  ride_h + 0.45, 0.25,  ride_h + 0.05),
        (cockpit_front_x,  0.45,  ride_h + 0.55, 0.40,  ride_h + 0.05),
        (-0.3,             0.48,  ride_h + 0.60, 0.42,  ride_h + 0.05),
        (cockpit_rear_x,   0.45,  ride_h + 0.55, 0.42,  ride_h + 0.05),
        (0.6,              0.42,  ride_h + 0.50, 0.38,  ride_h + 0.05),
        (engine_cover_x,   0.35,  ride_h + 0.65, 0.30,  ride_h + 0.10),
        (1.8,              0.25,  ride_h + 0.55, 0.20,  ride_h + 0.15),
        (rear_x,           0.15,  ride_h + 0.40, 0.10,  ride_h + 0.20),
    ]

    segments = 8  # Points per half-profile

    for i, (x, wt, ht, wb, hb) in enumerate(profiles):
        base_idx = len(verts)
        # Create half-oval profile, then mirror
        for j in range(segments + 1):
            t = j / segments  # 0 = bottom, 1 = top
            angle = t * math.pi  # 0 to pi
            w = wb + (wt - wb) * math.sin(angle)
            h = hb + (ht - hb) * (0.5 + 0.5 * math.sin(angle))
            # Right side
            verts.append((x, w, h))
        for j in range(segments, -1, -1):
            t = j / segments
            angle = t * math.pi
            w = wb + (wt - wb) * math.sin(angle)
            h = hb + (ht - hb) * (0.5 + 0.5 * math.sin(angle))
            # Left side
            verts.append((x, -w, h))

        # Connect to previous profile
        if i > 0:
            pts_per_profile = 2 * (segments + 1)
            prev_base = base_idx - pts_per_profile
            for j in range(pts_per_profile - 1):
                faces.append((
                    prev_base + j,
                    prev_base + j + 1,
                    base_idx + j + 1,
                    base_idx + j,
                ))
            # Close the loop
            faces.append((
                prev_base + pts_per_profile - 1,
                prev_base,
                base_idx,
                base_idx + pts_per_profile - 1,
            ))

    # Create mesh
    mesh = bpy.data.meshes.new("Monocoque")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Monocoque", mesh)
    bpy.context.collection.objects.link(obj)

    # Smooth shading
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)

    return obj


def create_front_wing():
    """
    Multi-element front wing.
    RB-style: full-width, aggressive angle, conditions flow for entire car.
    """
    wing_span = 2.000        # Full car width
    wing_chord = 0.350       # Main element chord
    wing_x = -2.5            # Front of car
    wing_z = params.ride_height + 0.05  # Just above ground

    angle_rad = math.radians(params.front_wing_angle)

    # Main element - NACA-like airfoil cross-section (simplified)
    verts = []
    faces = []
    n_span = 12
    n_chord = 10

    for i in range(n_span + 1):
        y = -wing_span / 2 + (wing_span / n_span) * i
        for j in range(n_chord + 1):
            t = j / n_chord
            # NACA 4-digit thickness distribution (simplified)
            x_local = t * wing_chord
            # Thickness: max 12% at 30% chord
            thickness = 0.12 * wing_chord * (
                2.969 * math.sqrt(t) - 1.260 * t - 3.516 * t**2
                + 2.843 * t**3 - 1.015 * t**4
            ) if t > 0 else 0

            # Apply angle of attack rotation
            x_rot = wing_x + x_local * math.cos(angle_rad) + thickness * math.sin(angle_rad)
            z_rot = wing_z - x_local * math.sin(angle_rad) + thickness * math.cos(angle_rad)
            verts.append((x_rot, y, z_rot))

    # Upper surface
    for i in range(n_span):
        for j in range(n_chord):
            idx = i * (n_chord + 1) + j
            faces.append((idx, idx + 1, idx + n_chord + 2, idx + n_chord + 1))

    # Lower surface (mirror thickness)
    offset = len(verts)
    for i in range(n_span + 1):
        y = -wing_span / 2 + (wing_span / n_span) * i
        for j in range(n_chord + 1):
            t = j / n_chord
            x_local = t * wing_chord
            thickness = 0.12 * wing_chord * (
                2.969 * math.sqrt(t) - 1.260 * t - 3.516 * t**2
                + 2.843 * t**3 - 1.015 * t**4
            ) if t > 0 else 0

            x_rot = wing_x + x_local * math.cos(angle_rad) - thickness * math.sin(angle_rad)
            z_rot = wing_z - x_local * math.sin(angle_rad) - thickness * math.cos(angle_rad)
            verts.append((x_rot, y, z_rot))

    for i in range(n_span):
        for j in range(n_chord):
            idx = offset + i * (n_chord + 1) + j
            faces.append((idx, idx + n_chord + 1, idx + n_chord + 2, idx + 1))

    # Front wing endplates
    endplate_height = 0.12
    endplate_chord = wing_chord * 1.2
    for side in [-1, 1]:
        ep_y = side * wing_span / 2
        base = len(verts)
        ep_verts = [
            (wing_x, ep_y, wing_z - 0.02),
            (wing_x + endplate_chord, ep_y, wing_z - 0.02),
            (wing_x + endplate_chord, ep_y, wing_z + endplate_height),
            (wing_x, ep_y, wing_z + endplate_height),
        ]
        verts.extend(ep_verts)
        faces.append((base, base + 1, base + 2, base + 3))

    mesh = bpy.data.meshes.new("FrontWing")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("FrontWing", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_rear_wing():
    """
    Rear wing assembly with main plane and DRS flap.
    RB-style: high efficiency, low drag when DRS open.
    """
    wing_span = 0.950         # Rear wing span (reg limited)
    main_chord = 0.250        # Main element chord
    flap_chord = 0.150        # DRS flap chord
    wing_x = 2.4              # Position
    wing_z = params.ride_height + 0.75  # Height

    angle_rad = math.radians(params.rear_wing_angle)
    flap_angle_rad = math.radians(params.rear_wing_angle + 8)  # Flap more aggressive

    verts = []
    faces = []
    n_span = 8
    n_chord = 8

    # Main element
    for i in range(n_span + 1):
        y = -wing_span / 2 + (wing_span / n_span) * i
        for j in range(n_chord + 1):
            t = j / n_chord
            x_local = t * main_chord
            thickness = 0.10 * main_chord * (
                2.969 * math.sqrt(t) - 1.260 * t - 3.516 * t**2
                + 2.843 * t**3 - 1.015 * t**4
            ) if t > 0 else 0

            x_rot = wing_x + x_local * math.cos(angle_rad)
            z_up = wing_z - x_local * math.sin(angle_rad) + thickness * math.cos(angle_rad)
            z_dn = wing_z - x_local * math.sin(angle_rad) - thickness * math.cos(angle_rad)
            # Upper surface
            verts.append((x_rot, y, z_up))

    for i in range(n_span):
        for j in range(n_chord):
            idx = i * (n_chord + 1) + j
            faces.append((idx, idx + 1, idx + n_chord + 2, idx + n_chord + 1))

    # DRS flap (above and behind main element)
    flap_x = wing_x + main_chord + 0.02
    flap_z = wing_z + 0.05
    offset = len(verts)

    for i in range(n_span + 1):
        y = -wing_span / 2 + (wing_span / n_span) * i
        for j in range(n_chord + 1):
            t = j / n_chord
            x_local = t * flap_chord
            thickness = 0.08 * flap_chord * (
                2.969 * math.sqrt(t) - 1.260 * t - 3.516 * t**2
                + 2.843 * t**3 - 1.015 * t**4
            ) if t > 0 else 0

            x_rot = flap_x + x_local * math.cos(flap_angle_rad)
            z_rot = flap_z - x_local * math.sin(flap_angle_rad) + thickness
            verts.append((x_rot, y, z_rot))

    for i in range(n_span):
        for j in range(n_chord):
            idx = offset + i * (n_chord + 1) + j
            faces.append((idx, idx + 1, idx + n_chord + 2, idx + n_chord + 1))

    # Endplates
    endplate_h = 0.20
    for side in [-1, 1]:
        ep_y = side * wing_span / 2
        base = len(verts)
        verts.extend([
            (wing_x - 0.05, ep_y, wing_z - 0.05),
            (flap_x + flap_chord + 0.05, ep_y, wing_z - 0.05),
            (flap_x + flap_chord + 0.05, ep_y, wing_z + endplate_h),
            (wing_x - 0.05, ep_y, wing_z + endplate_h),
        ])
        faces.append((base, base + 1, base + 2, base + 3))

    mesh = bpy.data.meshes.new("RearWing")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("RearWing", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_floor_and_diffuser():
    """
    Underbody floor with Venturi tunnels and diffuser.
    RB19-style: floor throat far rearward, high front floor, arched roof.
    This is the PRIMARY downforce generator in ground-effect F1 cars.
    """
    ride_h = params.ride_height
    diff_angle = math.radians(params.diffuser_angle)

    # Floor dimensions
    floor_front_x = -1.8
    floor_rear_x = 2.6
    tunnel_start_x = 0.5      # Where Venturi tunnels narrow
    diffuser_start_x = 1.8    # Where diffuser kicks up
    floor_length = floor_rear_x - floor_front_x

    verts = []
    faces = []

    n_length = 20
    n_width = 10

    for i in range(n_length + 1):
        t = i / n_length
        x = floor_front_x + t * floor_length

        for j in range(n_width + 1):
            s = j / n_width
            y = -HALF_FLOOR + s * FLOOR_WIDTH

            # Base height = ride height
            z = ride_h

            # Venturi tunnel profile (center section dips down)
            y_abs = abs(y)
            tunnel_width = 0.25  # Half-width of each tunnel
            tunnel_center = 0.45  # Y-center of tunnel

            if x > tunnel_start_x:
                tunnel_depth_factor = min(1.0, (x - tunnel_start_x) / (diffuser_start_x - tunnel_start_x))
                tunnel_depth = 0.015 * tunnel_depth_factor
                dist_from_center = abs(y_abs - tunnel_center)
                if dist_from_center < tunnel_width:
                    # Smooth tunnel profile
                    tunnel_shape = math.cos(dist_from_center / tunnel_width * math.pi / 2)
                    z -= tunnel_depth * tunnel_shape

            # Diffuser ramp
            if x > diffuser_start_x:
                diff_t = (x - diffuser_start_x) / (floor_rear_x - diffuser_start_x)
                diff_rise = diff_t * (floor_rear_x - diffuser_start_x) * math.tan(diff_angle)
                z += diff_rise

                # Diffuser expansion (wider at exit)
                expansion = 1.0 + 0.15 * diff_t
                y_expanded = y * expansion
                y = y_expanded

            verts.append((x, y, z))

    # Create faces
    for i in range(n_length):
        for j in range(n_width):
            idx = i * (n_width + 1) + j
            next_row = (i + 1) * (n_width + 1) + j
            faces.append((idx, idx + 1, next_row + 1, next_row))

    # Floor edge fences / vortex generators (simplified as thin plates)
    fence_height = 0.025
    for side in [-1, 1]:
        for fence_y in [0.35, 0.55, 0.70]:
            base = len(verts)
            fy = side * fence_y
            verts.extend([
                (tunnel_start_x, fy, ride_h),
                (diffuser_start_x, fy, ride_h),
                (diffuser_start_x, fy, ride_h - fence_height),
                (tunnel_start_x, fy, ride_h - fence_height),
            ])
            faces.append((base, base + 1, base + 2, base + 3))

    mesh = bpy.data.meshes.new("Floor_Diffuser")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Floor_Diffuser", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_sidepods():
    """
    Sidepods with RB20-style shark inlet and aggressive undercut.
    The undercut is critical for directing airflow to the diffuser.
    """
    ride_h = params.ride_height
    undercut = params.sidepod_undercut

    verts = []
    faces = []

    # Sidepod profiles at X stations
    sidepod_front_x = -0.4
    sidepod_rear_x = 1.6
    sidepod_length = sidepod_rear_x - sidepod_front_x

    n_length = 10

    for side in [-1, 1]:
        base_offset = len(verts)

        for i in range(n_length + 1):
            t = i / n_length
            x = sidepod_front_x + t * sidepod_length

            # Sidepod shape: inlet narrows, then undercut, then taper
            # Width profile (from chassis center)
            inner_y = side * 0.42  # Inner wall (near chassis)
            # Outer wall: starts narrow (inlet), expands, then tapers
            inlet_factor = 1.0 - 0.3 * math.exp(-((t - 0.1) / 0.15) ** 2)
            taper_factor = 1.0 - 0.4 * max(0, (t - 0.7)) / 0.3
            outer_y = side * (0.42 + 0.48 * inlet_factor * taper_factor)

            # Height profile
            top_z = ride_h + 0.35 + 0.20 * math.sin(t * math.pi)  # Rounded top
            # Undercut: bottom rises as we go rearward
            undercut_rise = undercut * t * t  # Quadratic undercut
            bottom_z = ride_h + 0.05 + undercut_rise

            # 6 points per cross-section: bottom-inner, mid-inner, top-inner,
            #                              top-outer, mid-outer, bottom-outer
            mid_z = (top_z + bottom_z) / 2
            pts = [
                (x, inner_y, bottom_z),
                (x, inner_y, mid_z),
                (x, inner_y, top_z),
                (x, outer_y, top_z),
                (x, outer_y, mid_z),
                (x, outer_y, bottom_z),
            ]
            verts.extend(pts)

        # Connect profiles
        pts_per = 6
        for i in range(n_length):
            for j in range(pts_per - 1):
                idx = base_offset + i * pts_per + j
                next_idx = base_offset + (i + 1) * pts_per + j
                faces.append((idx, idx + 1, next_idx + 1, next_idx))
            # Close the loop
            idx = base_offset + i * pts_per + pts_per - 1
            next_idx = base_offset + (i + 1) * pts_per + pts_per - 1
            faces.append((idx, base_offset + i * pts_per,
                         base_offset + (i + 1) * pts_per, next_idx))

    mesh = bpy.data.meshes.new("Sidepods")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Sidepods", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_wheels():
    """Create simplified wheel/tire assemblies."""
    wheels = []
    positions = [
        ("FL", -WHEELBASE / 2, TRACK_FRONT / 2, WHEEL_WIDTH_FRONT),
        ("FR", -WHEELBASE / 2, -TRACK_FRONT / 2, WHEEL_WIDTH_FRONT),
        ("RL", WHEELBASE / 2, TRACK_REAR / 2, WHEEL_WIDTH_REAR),
        ("RR", WHEELBASE / 2, -TRACK_REAR / 2, WHEEL_WIDTH_REAR),
    ]

    for name, x, y, width in positions:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=WHEEL_DIAMETER / 2,
            depth=width,
            location=(x, y, WHEEL_DIAMETER / 2),
            rotation=(math.pi / 2, 0, 0),
        )
        wheel = bpy.context.active_object
        wheel.name = f"Wheel_{name}"
        wheels.append(wheel)

    return wheels


def create_halo():
    """Create the Halo safety device."""
    ride_h = params.ride_height
    verts = []
    faces = []

    # Halo is a titanium structure above the cockpit
    halo_base_x = -0.6
    halo_front_x = -1.0
    halo_top_z = ride_h + 0.80
    halo_base_z = ride_h + 0.55
    halo_width = 0.38

    n_arc = 16
    tube_radius = 0.025

    # Generate halo arc path
    path_points = []
    for i in range(n_arc + 1):
        t = i / n_arc
        # Arc from base, forward and up, then back
        angle = t * math.pi
        x = halo_base_x + (halo_front_x - halo_base_x) * math.sin(angle)
        z = halo_base_z + (halo_top_z - halo_base_z) * math.sin(angle)
        path_points.append((x, 0, z))

    # Create tube along path
    n_tube = 8
    for i, (px, py, pz) in enumerate(path_points):
        base = len(verts)
        for j in range(n_tube):
            angle = 2 * math.pi * j / n_tube
            vy = py + tube_radius * math.cos(angle)
            vz = pz + tube_radius * math.sin(angle)
            verts.append((px, vy, vz))

        if i > 0:
            prev_base = base - n_tube
            for j in range(n_tube):
                j_next = (j + 1) % n_tube
                faces.append((
                    prev_base + j,
                    prev_base + j_next,
                    base + j_next,
                    base + j,
                ))

    # Side arms (Y-branches from center to cockpit sides)
    for side in [-1, 1]:
        arm_points = [
            (halo_base_x + 0.15, 0, halo_top_z - 0.05),
            (halo_base_x + 0.05, side * halo_width / 2, halo_base_z + 0.02),
        ]
        for k, (ax, ay, az) in enumerate(arm_points):
            base = len(verts)
            for j in range(n_tube):
                angle = 2 * math.pi * j / n_tube
                vy = ay + tube_radius * math.cos(angle)
                vz = az + tube_radius * math.sin(angle)
                verts.append((ax, vy, vz))

            if k > 0:
                prev_base = base - n_tube
                for j in range(n_tube):
                    j_next = (j + 1) % n_tube
                    faces.append((
                        prev_base + j,
                        prev_base + j_next,
                        base + j_next,
                        base + j,
                    ))

    mesh = bpy.data.meshes.new("Halo")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Halo", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# =============================================================================
# Export Functions
# =============================================================================

def export_stl(output_dir):
    """Export each component as individual STL and one combined STL."""
    os.makedirs(output_dir, exist_ok=True)

    # Detect STL export API (changed in Blender 5.x)
    has_new_api = hasattr(bpy.ops.wm, "stl_export")

    # Export individual components
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            filepath = os.path.join(output_dir, f"{obj.name}.stl")
            if has_new_api:
                bpy.ops.wm.stl_export(
                    filepath=filepath,
                    export_selected_objects=True,
                    ascii_format=False,
                )
            else:
                bpy.ops.export_mesh.stl(
                    filepath=filepath,
                    use_selection=True,
                    ascii=False,
                )
            print(f"  Exported: {filepath}")

    # Export combined
    bpy.ops.object.select_all(action="SELECT")
    combined_path = os.path.join(output_dir, "f1_car_combined.stl")
    if has_new_api:
        bpy.ops.wm.stl_export(
            filepath=combined_path,
            export_selected_objects=True,
            ascii_format=False,
        )
    else:
        bpy.ops.export_mesh.stl(
            filepath=combined_path,
            use_selection=True,
            ascii=False,
        )
    print(f"  Exported combined: {combined_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAS_BLENDER:
        print("Cannot generate geometry without Blender.")
        print("Run: blender --background --python f1_car_generator.py")
        print(f"\nParameters: {vars(params)}")
        return

    print("=" * 60)
    print("  F1 Car Parametric Geometry Generator")
    print("  Based on Red Bull RB19/RB20 design philosophy")
    print("=" * 60)
    print(f"  Ride height:       {params.ride_height * 1000:.1f} mm")
    print(f"  Front wing angle:  {params.front_wing_angle:.1f} deg")
    print(f"  Rear wing angle:   {params.rear_wing_angle:.1f} deg")
    print(f"  Diffuser angle:    {params.diffuser_angle:.1f} deg")
    print(f"  Sidepod undercut:  {params.sidepod_undercut * 1000:.1f} mm")
    print("=" * 60)

    clear_scene()

    # Create materials
    mat_carbon = create_material("Carbon_Fiber", (0.05, 0.05, 0.08, 1.0))
    mat_blue = create_material("RB_Blue", (0.0, 0.02, 0.25, 1.0))
    mat_yellow = create_material("RB_Yellow", (1.0, 0.85, 0.0, 1.0))
    mat_rubber = create_material("Tire_Rubber", (0.1, 0.1, 0.1, 1.0))
    mat_titanium = create_material("Titanium", (0.6, 0.6, 0.55, 1.0))

    # Generate components
    print("\nGenerating components...")

    print("  [1/6] Monocoque chassis...")
    monocoque = create_monocoque()
    assign_material(monocoque, mat_blue)

    print("  [2/6] Front wing...")
    front_wing = create_front_wing()
    assign_material(front_wing, mat_carbon)

    print("  [3/6] Rear wing...")
    rear_wing = create_rear_wing()
    assign_material(rear_wing, mat_carbon)

    print("  [4/6] Floor & diffuser (Venturi tunnels)...")
    floor = create_floor_and_diffuser()
    assign_material(floor, mat_carbon)

    print("  [5/6] Sidepods...")
    sidepods = create_sidepods()
    assign_material(sidepods, mat_blue)

    print("  [6/6] Wheels & Halo...")
    wheels = create_wheels()
    for w in wheels:
        assign_material(w, mat_rubber)

    halo = create_halo()
    assign_material(halo, mat_titanium)

    # Set up camera and lighting for renders
    bpy.ops.object.camera_add(location=(6, -4, 3))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(65), 0, math.radians(55))
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    # Export
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    output_dir = params.output_dir or os.path.join(project_dir, "openfoam", "f1_baseline", "constant", "triSurface")
    print(f"\nExporting STL files to: {output_dir}")
    export_stl(output_dir)

    # Save .blend file
    blend_path = params.output_blend or os.path.join(project_dir, "blender", "f1_baseline.blend")
    print(f"Saving Blender file: {blend_path}")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print("\nDone! Components generated:")
    print(f"  - Monocoque: {len(monocoque.data.vertices)} verts, {len(monocoque.data.polygons)} faces")
    print(f"  - Front wing: {len(front_wing.data.vertices)} verts")
    print(f"  - Rear wing: {len(rear_wing.data.vertices)} verts")
    print(f"  - Floor/diffuser: {len(floor.data.vertices)} verts")
    print(f"  - Sidepods: {len(sidepods.data.vertices)} verts")
    print(f"  - Halo: {len(halo.data.vertices)} verts")
    print(f"  - 4 wheels")
    print(f"\nTotal mesh objects: {len([o for o in bpy.context.scene.objects if o.type == 'MESH'])}")


if __name__ == "__main__":
    main()
