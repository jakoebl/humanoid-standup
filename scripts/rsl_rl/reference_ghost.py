"""Visual-only, translucent reference overlay for tracking playback."""
import gymnasium as gym


def resolve_link_paths(prims, body_names):
    """Resolve rigid-body links, excluding visual/collision prims with the same name."""
    candidates = {}
    for prim in prims:
        if "PhysicsRigidBodyAPI" not in prim.GetAppliedSchemas():
            continue
        candidates.setdefault(prim.GetName(), []).append(str(prim.GetPath()))
    paths = []
    for name in body_names:
        matches = candidates.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"Expected one robot link named {name!r}, found {matches}")
        paths.append(matches[0])
    return paths


def flattened_visual_layer(stage):
    """Expand instances on a detached stage so subtree copies have no prototype dependencies."""
    from pxr import Usd

    detached = Usd.Stage.Open(stage.Flatten())
    while True:
        instances = [prim for prim in detached.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    return detached.Flatten()


def remove_collision_subtrees(stage, prim):
    """Collect paths before removal so deleting a parent cannot expire traversed prims."""
    from pxr import Usd

    paths = [child.GetPath() for child in Usd.PrimRange(prim) if child.GetName() == "collisions"]
    for path in reversed(paths):
        stage.RemovePrim(path)


class ReferenceGhost(gym.Wrapper):
    """Copy robot link visuals and animate them from the prepared reference poses."""

    def __init__(self, env):
        super().__init__(env)
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        self.Gf = Gf
        self.motion = env.unwrapped.command_manager.get_term("motion")
        self.stage = omni.usd.get_context().get_stage()
        root = "/Visuals/ReferenceGhost"
        UsdGeom.Xform.Define(self.stage, root)
        material = UsdShade.Material.Define(self.stage, root + "/Material")
        shader = UsdShade.Shader.Define(self.stage, root + "/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.8, 1.0))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.25)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        # USD Flatten preserves instances; expand them before copying only link subtrees.
        flattened = flattened_visual_layer(self.stage)
        robot = env.unwrapped.scene["robot"]
        scene = env.unwrapped.scene
        robot_path = robot.cfg.prim_path.replace("{ENV_REGEX_NS}", scene.env_prim_paths[0])
        robot_path = robot_path.replace(scene.env_regex_ns, scene.env_prim_paths[0])
        robot_prim = self.stage.GetPrimAtPath(robot_path)
        if not robot_prim.IsValid():
            raise ValueError(f"Cannot find robot visuals: {robot_path}")
        sources = resolve_link_paths(Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()), robot.body_names)
        self.links = []
        mesh_count = 0
        for name, source in zip(robot.body_names, sources):
            destination = f"{root}/{name}"
            Sdf.CopySpec(flattened, source, self.stage.GetRootLayer(), destination)
            prim = self.stage.GetPrimAtPath(destination)
            # Collision subtrees are never part of the visible reference.
            remove_collision_subtrees(self.stage, prim)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            # No physics schemas or joints may survive on the reference copy.
            descendants = list(Usd.PrimRange(prim))
            for child in reversed(descendants):
                if "Joint" in child.GetTypeName():
                    self.stage.RemovePrim(child.GetPath())
                    continue
                schemas = [schema for schema in child.GetAppliedSchemas()
                           if "physics" not in schema.lower() and "physx" not in schema.lower()]
                child.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))
                if child.IsA(UsdGeom.Imageable):
                    imageable = UsdGeom.Imageable(child)
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                    imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                if child.IsA(UsdGeom.Mesh):
                    mesh_count += 1
                if child.IsA(UsdGeom.Gprim):
                    UsdShade.MaterialBindingAPI.Apply(child).Bind(
                        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            transform = UsdGeom.Xformable(prim)
            transform.ClearXformOpOrder()
            transform.SetResetXformStack(True)
            translation = transform.AddTranslateOp()
            orientation = transform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            self.links.append((translation, orientation))
        if mesh_count == 0:
            raise RuntimeError("Reference ghost contains no renderable meshes.")
        self.update_reference()
        print(f"[INFO] Reference ghost: {len(self.links)} links, {mesh_count} visible meshes; USD rendering.",
              flush=True)

    def update_reference(self):
        frame = int(self.motion.time_steps[0].item())
        origin = self.env.unwrapped.scene.env_origins[0]
        positions = (self.motion.motion._body_pos_w[frame] + origin).cpu().tolist()
        quaternions = self.motion.motion._body_quat_w[frame].cpu().tolist()
        for (translation, orientation), position, quaternion in zip(self.links, positions, quaternions):
            translation.Set(self.Gf.Vec3d(*position))
            orientation.Set(self.Gf.Quatd(quaternion[0], self.Gf.Vec3d(*quaternion[1:])))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.update_reference()
        return result

    def step(self, action):
        result = self.env.step(action)
        self.update_reference()
        # RecordVideo reads the RGB annotator after this wrapper returns.
        self.env.unwrapped.sim.render()
        return result
