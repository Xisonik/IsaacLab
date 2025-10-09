import json
import os
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils

class AssetManager:
    """Спавнит объекты только в /World/envs/env_0, возвращает их пути и счётчики."""
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        self.items = cfg["objects"]                       # список описаний объектов
        self.prim_paths = {}                              # name -> [prim_path_env0_i, ...]
        self.counts = {}                                  # name -> count

    def _abs_paths(self, usd_paths):
        root = os.getcwd()
        return [os.path.join(root, "source/isaaclab_assets/data/aloha_assets", p) for p in usd_paths]

    def _rigid_props_for(self, types: str):
        if types == "static_obstacle":
            return sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True)
        elif types == "movable_obstacle":
            return sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True,  disable_gravity=True)
        else:
            return sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=True)

    def spawn_assets_in_env0(self):
        """Создаёт все объекты только под /World/envs/env_0/... (без регистрации в scene)."""
        self.prim_paths.clear()
        self.counts.clear()

        row = 0
        for obj in self.items:
            name  = obj["name"]
            types = obj["type"]
            if "info" in types:
                continue

            count     = int(obj["count"])
            usd_paths = self._abs_paths(obj["usd_paths"])
            self.prim_paths[name] = []
            self.counts[name] = count

            default_rot = (1.0, 0.0, 0.0, 0.0)
            if name == "bowl":
                default_rot = (0.0, 0.7071, 0.0, 0.7071)

            for i in range(count):
                if  "movable_obstacle" in types or  "static_obstacle" in types:
                    prim_path = f"/World/envs/env_0/obstacles/{name}_{i}"
                    spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=usd_paths[0],
                        rigid_props=self._rigid_props_for(types),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        activate_contact_sensors=False,
                    )
                else:
                    prim_path = f"/World/envs/env_0/{name}_{i}"
                    spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=usd_paths[0],
                        rigid_props=self._rigid_props_for(types),
                        activate_contact_sensors=False,
                    )

                RigidObject(
                    RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=spawn_cfg,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(5.0 + i, 6.0 + row, 0.4),
                            rot=default_rot,
                        ),
                    )
                )
                self.prim_paths[name].append(prim_path)

            row += 1

        return self.prim_paths, self.counts
