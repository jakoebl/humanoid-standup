"""USD regression: link-only copies retain nested instanced meshes."""
import ast
from pathlib import Path
import unittest

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "Requires pxr (Isaac Sim Python or usd-core)")
class ReferenceGhostUsdTest(unittest.TestCase):
    def test_collision_removal_does_not_access_expired_children(self):
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/Ghost/pelvis").GetPrim()
        UsdGeom.Xform.Define(stage, "/Ghost/pelvis/collisions/pelvis_contour_link")
        UsdGeom.Mesh.Define(stage, "/Ghost/pelvis/collisions/pelvis_contour_link/mesh")
        UsdGeom.Xform.Define(stage, "/Ghost/pelvis/collisions/nested/collisions/child")
        UsdGeom.Mesh.Define(stage, "/Ghost/pelvis/visuals/pelvis")
        path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/reference_ghost.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        namespace["remove_collision_subtrees"](stage, body)
        self.assertFalse(stage.GetPrimAtPath("/Ghost/pelvis/collisions").IsValid())
        self.assertTrue(stage.GetPrimAtPath("/Ghost/pelvis/visuals/pelvis").IsValid())
        self.assertTrue(body.IsValid())

    def test_subtree_copy_contains_meshes_without_prototype_references(self):
        source = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(source, "/Inner")
        UsdGeom.Mesh.Define(source, "/Inner/pelvis")
        UsdGeom.Xform.Define(source, "/Outer")
        nested = source.DefinePrim("/Outer/nested", "Xform")
        nested.GetReferences().AddInternalReference("/Inner")
        nested.SetInstanceable(True)
        body = UsdGeom.Xform.Define(source, "/Robot/pelvis").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        visuals = source.DefinePrim("/Robot/pelvis/visuals", "Xform")
        visuals.GetReferences().AddInternalReference("/Outer")
        visuals.SetInstanceable(True)
        path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/reference_ghost.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        paths = namespace["resolve_link_paths"](
            Usd.PrimRange(source.GetPrimAtPath("/Robot"), Usd.TraverseInstanceProxies()), ["pelvis"])
        flattened = namespace["flattened_visual_layer"](source)
        destination = Usd.Stage.CreateInMemory()
        destination.DefinePrim("/Ghost", "Xform")
        self.assertTrue(Sdf.CopySpec(flattened, paths[0], destination.GetRootLayer(), "/Ghost/pelvis"))
        mesh = destination.GetPrimAtPath("/Ghost/pelvis/visuals/nested/pelvis")
        self.assertTrue(mesh.IsValid())
        self.assertTrue(mesh.IsA(UsdGeom.Mesh))
        self.assertFalse(any(prim.IsInstance() for prim in destination.Traverse()))
        self.assertNotIn("Flattened_Prototype", destination.GetRootLayer().ExportToString())
        self.assertTrue(visuals.IsInstance())  # Source physics asset remains untouched.


if __name__ == "__main__":
    unittest.main()
