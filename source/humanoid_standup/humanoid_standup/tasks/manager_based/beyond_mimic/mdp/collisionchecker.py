"""Check reset poses against the floor using the spawned robot's collision geometry."""

import math
import re
from itertools import product

import numpy as np
import omni.usd
import torch
from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.utils.math import quat_apply


class CollisionChecker:
    """Cache body-local collision shapes and test their lowest points against the floor."""

    def __init__(self, robot, env_origins, tolerance=1e-4):
        # Store reset settings and the flat floor height for each environment.
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Floor penetration tolerance must be finite and nonnegative.")
        self.robot = robot
        self.floor_heights = env_origins[:, 2]
        self.tolerance = tolerance
        self.shapes = []

        # Locate the first robot clone, starting from its articulation root.
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(robot.root_physx_view.prim_paths[0])
        while root and not re.fullmatch(robot.cfg.prim_path, str(root.GetPath())):
            root = root.GetParent()
        if not root:
            raise RuntimeError("Cannot locate robot geometry for floor checking.")

        # Include instanced collision shapes and map their owning rigid bodies.
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        bodies = {str(prim.GetPath()): robot.body_names.index(prim.GetName()) for prim in prims
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in robot.body_names}
        cache = UsdGeom.XformCache()
        for prim in prims:
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            body = prim
            while body and str(body.GetPath()) not in bodies:
                body = body.GetParent()
            if not body:
                raise RuntimeError(f"Cannot resolve collision body for {prim.GetPath()}")

            # Bake each collider's offset and scale into its rigid-body frame.
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim) *
                                cache.GetLocalToWorldTransform(body).GetInverse())
            radius = 0.
            if prim.IsA(UsdGeom.Sphere):
                scales = np.linalg.norm(matrix[:3, :3], axis=1)
                if not np.allclose(scales, scales[0]):
                    raise RuntimeError(f"Unsupported nonuniform sphere scale: {prim.GetPath()}")
                points = np.zeros((1, 3))
                radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * scales[0]
            elif prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            elif prim.IsA(UsdGeom.Cube):
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                points = np.asarray(list(product((-size, size), repeat=3)))
            else:
                raise RuntimeError(f"Unsupported floor collision shape: {prim.GetPath()} ({prim.GetTypeName()})")
            points = points @ matrix[:3, :3] + matrix[3, :3]
            self.shapes.append({"body": bodies[str(body.GetPath())], "radius": radius,
                                "points": torch.tensor(points, dtype=torch.float32, device=robot.device)})
        if not self.shapes:
            raise RuntimeError("No collision shapes found for floor checking.")

    def invalid_envs(self, env_ids):
        # Read poses only for environments still being reset.
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.robot.device)
        positions = self.robot.data.body_link_pos_w[ids]
        rotations = self.robot.data.body_link_quat_w[ids]
        if not torch.isfinite(positions).all() or not torch.isfinite(rotations).all():
            raise RuntimeError("Nonfinite body transforms during floor checking.")

        # Compare every collider's lowest world point with its environment's floor.
        invalid = torch.zeros(len(ids), dtype=torch.bool, device=self.robot.device)
        floor = self.floor_heights[ids] - self.tolerance
        for shape in self.shapes:
            points = shape["points"].expand(len(ids), -1, -1)
            quaternions = rotations[:, shape["body"], None, :].expand(-1, points.shape[1], -1)
            rotated = quat_apply(quaternions, points)
            lowest = rotated[:, :, 2].amin(dim=1) + positions[:, shape["body"], 2] - shape["radius"]
            invalid |= lowest < floor
        return ids[invalid].cpu().tolist()
