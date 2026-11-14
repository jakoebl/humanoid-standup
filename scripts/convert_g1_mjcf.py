"""Convert the humanoid-mrl G1 MJCF into the USD used by the BeyondMimic task.

The MJCF has simple collision capsules on the forearms and hands, which the
rev 1.0 Isaac Sim asset lacks. The massless ``*_foot_contact_point`` bodies are
dropped because the reference motions do not contain them, and the ankle
tendons because PhysX imports them with a negligible limit stiffness. Isaac Lab's importer
nests every body under a grouping ``pelvis`` prim and adds an empty
``worldBody`` articulation root; both are undone so bodies sit at
``/g1_29dof/<body>`` like in the stock Isaac Lab G1.
"""

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the G1 MJCF to a flat USD.")
parser.add_argument("mjcf", type=Path, help="Path to humanoid-mrl's mujoco/g1/g1_29dof.xml.")
parser.add_argument("output_dir", type=Path, help="Directory for g1_29dof.usd and its layers.")
args = parser.parse_args()

app = AppLauncher(headless=True).app

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from pxr import Sdf  # noqa: E402

ROOT = Sdf.Path("/g1_29dof")
GROUP = ROOT.AppendChild("pelvis")
TMP = ROOT.AppendChild("_group")


def strip_mjcf(src, dst):
    tree = ET.parse(src)
    compiler = tree.find("compiler")
    # The importer resolves meshdir relative to the MJCF file, even when it is absolute.
    compiler.set("meshdir", os.path.relpath(src.parent / compiler.get("meshdir"), dst.parent) + "/")
    for parent in tree.iter():
        for body in parent.findall("body"):
            if body.get("name").endswith("_foot_contact_point"):
                parent.remove(body)
    tree.getroot().remove(tree.find("tendon"))
    tree.write(dst)


def retarget(path):
    path = Sdf.Path(str(path).replace(str(GROUP) + "/", str(TMP) + "/"))
    return path.ReplacePrefix(TMP, ROOT) if path.HasPrefix(TMP) else path


def flatten(layer):
    root = layer.GetPrimAtPath(ROOT)
    if root is None:
        return
    if "worldBody" in root.nameChildren:
        del root.nameChildren["worldBody"]
    if layer.GetPrimAtPath(GROUP) is not None:
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(GROUP, TMP)
        layer.Apply(edit)
        for child in list(layer.GetPrimAtPath(TMP).nameChildren):
            Sdf.CopySpec(layer, TMP.AppendChild(child.name), layer, ROOT.AppendChild(child.name))
        del root.nameChildren[TMP.name]

    def fix(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec):
            for items in ("explicitItems", "addedItems", "prependedItems", "appendedItems", "deletedItems", "orderedItems"):
                targets = list(getattr(spec.targetPathList, items))
                if targets:
                    setattr(spec.targetPathList, items, [retarget(t) for t in targets])

    layer.Traverse(Sdf.Path.absoluteRootPath, fix)
    layer.Save()


with tempfile.TemporaryDirectory() as tmp:
    mjcf = Path(tmp) / "g1_29dof.xml"
    strip_mjcf(args.mjcf.resolve(), mjcf)
    MjcfConverter(MjcfConverterCfg(
        asset_path=str(mjcf), usd_dir=str(args.output_dir.resolve()), usd_file_name="g1_29dof.usd",
        force_usd_conversion=True, fix_base=False, import_sites=False,
    ))
for path in (args.output_dir / "configuration").glob("*.usd"):
    flatten(Sdf.Layer.FindOrOpen(str(path)))
app.close()
