from __future__ import annotations

import hashlib
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class LoadedModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_idx: np.ndarray
    qvel_idx: np.ndarray


@dataclass(frozen=True)
class ModelLoadOptions:
    payload_mass: float = 0.0
    payload_radius: float = 0.04
    payload_rgba: tuple[float, float, float, float] = (0.2, 0.6, 0.9, 0.9)


def load_model(model_path: Path, *, options: ModelLoadOptions | None = None) -> LoadedModel:
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if model_path.suffix.lower() == ".urdf":
        mjcf_path = _urdf_to_actuated_mjcf(model_path, options=options)
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        if model.nu == 0:
            raise ValueError(
                f"Model has no actuators (nu=0): {model_path}. "
                "Use torque motors or pass an actuated MJCF."
            )

    data = mujoco.MjData(model)
    qpos_idx, qvel_idx = _actuated_state_indices(model)
    return LoadedModel(model=model, data=data, qpos_idx=qpos_idx, qvel_idx=qvel_idx)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_dir() -> Path:
    d = _repo_root() / ".cache" / "mujoco"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _urdf_workdir(urdf_path: Path) -> Path:
    h = _sha256(urdf_path)[:12]
    d = _cache_dir() / f"{urdf_path.stem}.{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prepare_urdf_mesh_paths(urdf_path: Path) -> Path:
    """
    MuJoCo's URDF loader often resolves mesh references by basename within the URDF
    directory. To make loading robust, we stage all referenced mesh files next to a
    rewritten URDF inside a per-hash cache directory.
    """
    workdir = _urdf_workdir(urdf_path)
    prepared = workdir / "prepared.urdf"
    if prepared.exists():
        return prepared

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    urdf_dir = urdf_path.parent
    package_root = urdf_dir.parent

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue

        resolved = _resolve_mesh_path(filename, package_root=package_root, urdf_dir=urdf_dir)
        if resolved is None:
            continue

        if resolved.suffix.lower() == ".dae":
            staged = _convert_dae_to_stl(resolved, workdir=workdir)
        else:
            staged = workdir / resolved.name
            if not staged.exists():
                shutil.copy2(resolved, staged)

        mesh.set("filename", staged.name)

    tree.write(prepared, encoding="utf-8", xml_declaration=True)
    return prepared


def _resolve_mesh_path(
    filename: str, *, package_root: Path, urdf_dir: Path
) -> Path | None:
    if filename.startswith("package://"):
        rest = filename.removeprefix("package://")

        # Common ROS layout in this repo:
        #   urdf/irb2400.urdf uses package://visual/... and package://collision/...
        # while meshes live under meshes/visual and meshes/collision.
        if rest.startswith(("visual/", "collision/")):
            candidate = package_root / "meshes" / rest
            if candidate.exists():
                return candidate.resolve()

        candidate = package_root / rest
        if candidate.exists():
            return candidate.resolve()

        candidate = urdf_dir / rest
        if candidate.exists():
            return candidate.resolve()

        return None

    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    candidate = (urdf_dir / filename).resolve()
    if candidate.exists():
        return candidate
    return None


def _urdf_to_actuated_mjcf(urdf_path: Path, *, options: ModelLoadOptions | None) -> Path:
    prepared_urdf = _prepare_urdf_mesh_paths(urdf_path)
    workdir = prepared_urdf.parent

    mjcf_raw = workdir / "raw.xml"
    suffix = ""
    if options is not None and options.payload_mass > 0:
        suffix = f".payload{options.payload_mass:.3f}"
    mjcf_act = workdir / f"actuated{suffix}.xml"
    if mjcf_act.exists():
        return mjcf_act

    model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
    mujoco.mj_saveLastXML(str(mjcf_raw), model)

    _patch_mjcf_for_torque_control(mjcf_raw, mjcf_act, options=options)
    return mjcf_act


def _patch_mjcf_for_torque_control(
    src_mjcf: Path, dst_mjcf: Path, *, options: ModelLoadOptions | None
) -> None:
    tree = ET.parse(src_mjcf)
    root = tree.getroot()

    _ensure_reasonable_joint_defaults(root)
    _patch_visual_meshes_inplace(root, asset_dir=src_mjcf.parent)
    _maybe_add_payload(root, options=options)

    # URDF import often inserts a floating base (<freejoint/>). This project assumes
    # a fixed-base industrial arm.
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            freejoint = top_body.find("freejoint")
            if freejoint is not None:
                top_body.remove(freejoint)

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    else:
        for child in list(actuator):
            actuator.remove(child)

    joint_names: list[str] = []
    for joint in root.findall(".//joint"):
        joint_type = joint.get("type", "hinge")
        if joint_type not in ("hinge", "slide"):
            continue
        name = joint.get("name")
        if name:
            joint_names.append(name)

    if not joint_names:
        raise ValueError(f"No hinge/slide joints found in {src_mjcf}")

    for name in joint_names:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"m_{name}",
                "joint": name,
                "ctrllimited": "true",
                "ctrlrange": "-5000 5000",
            },
        )

    tree.write(dst_mjcf, encoding="utf-8", xml_declaration=True)


def _ensure_reasonable_joint_defaults(root: ET.Element) -> None:
    """
    URDF-only meshes (no inertials) can produce extremely small wrist inertias after import,
    which makes torque control numerically unstable. Add conservative joint damping and
    armature as a stabilizing baseline for controller development.
    """
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")

    joint = default.find("joint")
    if joint is None:
        joint = ET.SubElement(default, "joint")

    joint.attrib.setdefault("damping", "1.0")
    joint.attrib.setdefault("armature", "0.05")


def _patch_visual_meshes_inplace(root: ET.Element, *, asset_dir: Path) -> None:
    """
    URDF importer often keeps only collision meshes. If we staged converted visual meshes
    next to the URDF (e.g. `link_1__visual.stl`), switch the MJCF mesh assets to those
    files. This improves rendering quality in the viewer.

    Note: This also affects collision/inertia if those depend on meshes. For the current
    milestone we disable contacts on all geoms to keep simulation stable.
    """
    asset = root.find("asset")
    if asset is None:
        return

    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file", "")
        if not file_attr:
            continue

        # base_link.stl -> base_link__visual.stl
        candidate = asset_dir / f"{Path(file_attr).stem}__visual.stl"

        # link_2_whole.stl -> link_2__visual.stl
        if not candidate.exists() and Path(file_attr).stem.endswith("_whole"):
            base = Path(file_attr).stem.removesuffix("_whole")
            candidate = asset_dir / f"{base}__visual.stl"

        if candidate.exists():
            mesh.set("file", candidate.name)
            mesh.set("content_type", "model/stl")

    # Disable contacts on imported geoms (visual-only milestone).
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.findall(".//geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _maybe_add_payload(root: ET.Element, *, options: ModelLoadOptions | None) -> None:
    if options is None or options.payload_mass <= 0:
        return

    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    # Heuristic: put payload on the deepest body in the kinematic chain.
    def deepest_body(b: ET.Element) -> ET.Element:
        cur = b
        while True:
            children = cur.findall("body")
            if not children:
                return cur
            cur = children[0]

    top_bodies = worldbody.findall("body")
    if not top_bodies:
        return
    tip = deepest_body(top_bodies[0])

    payload = ET.SubElement(tip, "body", {"name": "payload", "pos": "0 0 0"})
    r = float(options.payload_radius)
    rgba = " ".join(str(float(x)) for x in options.payload_rgba)
    ET.SubElement(
        payload,
        "geom",
        {
            "type": "sphere",
            "size": f"{r}",
            "mass": f"{float(options.payload_mass)}",
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _actuated_state_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    if model.nu == 0:
        raise ValueError("Model has no actuators (nu=0).")

    qpos_idx = np.zeros(model.nu, dtype=int)
    qvel_idx = np.zeros(model.nu, dtype=int)
    for a in range(model.nu):
        joint_id = int(model.actuator_trnid[a, 0])
        qpos_idx[a] = int(model.jnt_qposadr[joint_id])
        qvel_idx[a] = int(model.jnt_dofadr[joint_id])
    return qpos_idx, qvel_idx


def _find_blender() -> str | None:
    import shutil as _shutil

    path = _shutil.which("blender")
    if path:
        return path

    candidate = Path.home() / ".local" / "bin" / "blender"
    if candidate.exists():
        return str(candidate)

    return None


def _convert_dae_to_stl(src_dae: Path, *, workdir: Path) -> Path:
    """
    MuJoCo does not support Collada (.dae) meshes directly; convert to STL via Blender.
    Output is cached in the per-URDF workdir to avoid repeated conversions.
    """
    dst = workdir / f"{src_dae.stem}__visual.stl"
    if dst.exists():
        return dst

    blender = _find_blender()
    if blender is None:
        raise RuntimeError(
            "Need Blender to convert .dae -> .stl for visual meshes, but blender was not found in PATH."
        )

    script = workdir / "_blender_convert_dae_to_stl.py"
    script.write_text(
        "\n".join(
            [
                "import bpy",
                "import sys",
                "",
                "argv = sys.argv",
                "argv = argv[argv.index('--')+1:] if '--' in argv else []",
                "src = argv[0]",
                "dst = argv[1]",
                "",
                "# Reset to empty scene",
                "bpy.ops.wm.read_factory_settings(use_empty=True)",
                "",
                "# Import collada",
                "bpy.ops.wm.collada_import(filepath=src)",
                "",
                "# Ensure objects are selected for export",
                "for obj in bpy.context.scene.objects:",
                "    obj.select_set(True)",
                "",
                "# Export STL (binary) - Blender 4.x uses bpy.ops.wm.stl_export",
                "if hasattr(bpy.ops.wm, 'stl_export'):",
                "    bpy.ops.wm.stl_export(filepath=dst, export_selected_objects=True)",
                "else:",
                "    bpy.ops.export_mesh.stl(filepath=dst, use_selection=True, ascii=False)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [blender, "-b", "-P", str(script), "--", str(src_dae), str(dst)],
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(
            "Blender mesh conversion failed.\n"
            f"cmd: {proc.args}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    return dst
