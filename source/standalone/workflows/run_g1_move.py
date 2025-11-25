import argparse
import torch
import os

from isaaclab.app import AppLauncher

# Argumente parsen
parser = argparse.ArgumentParser(description="Unitree G1 Arm Waving")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# App starten
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports (erst NACH App-Start)
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# PFAD ZUR USD DATEI
USD_PATH = "C:/Users/tshed/Documents/WIP/IsaacLab/resources/g1/g1_29dof/g1_29dof.usd"

@configclass
class UnitreeG1Cfg(ArticulationCfg):
    """Konfiguration für den G1 Roboter."""
    
    # 1. Wo soll der Roboter spawnen?
    prim_path = "/World/Robot"
    
    # 2. Welche Datei laden wir?
    spawn = sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
    )
    
    # 3. Start-Status (Position etwas höher, damit er nicht im Boden steckt)
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
    )
    
    # 4. Basis fixieren (Kein Umfallen)
    fix_root_link = True
    
    # 5. Aktuatoren definieren
    actuators = {
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=200.0,
            damping=10.0,
        ),
    }

def main():
    # Simulation Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Szene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Roboter laden
    robot_cfg = UnitreeG1Cfg()
    robot = Articulation(cfg=robot_cfg)
    
    # --- HIER WAR DER TIPPFEHLER KORRIGIERT ---
    scene.articulations["g1"] = robot
    # ------------------------------------------

    # Start
    sim.reset()
    print(f"[INFO] G1 geladen. Anzahl Gelenke: {robot.num_joints}")
    
    # Arm Indizes finden
    arm_indices = [
        i for i, n in enumerate(robot.joint_names) 
        if "right" in n and ("shoulder" in n or "elbow" in n)
    ]
    print(f"[INFO] Arm Gelenke: {arm_indices}")

    # Startposition merken
    default_pos = robot.data.default_joint_pos.clone()
    targets = default_pos.clone()
    
    sim_time = 0.0
    while simulation_app.is_running():
        sim_time += sim_cfg.dt
        
        # Winken
        if len(arm_indices) > 0:
            shoulder_idx = arm_indices[0]
            targets[:, shoulder_idx] = default_pos[:, shoulder_idx] + 1.0 * torch.sin(torch.tensor(sim_time * 2.0))
            
            if len(arm_indices) > 2:
                elbow_idx = arm_indices[2]
                targets[:, elbow_idx] = default_pos[:, elbow_idx] + 0.5 * torch.sin(torch.tensor(sim_time * 3.0))

        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        scene.update(dt=sim_cfg.dt)
    # Simulation Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Szene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Roboter laden
    robot_cfg = UnitreeG1Cfg()
    robot = Articulation(cfg=robot_cfg)
    scene.articulatons["g1"] = robot

    # Start
    sim.reset()
    print(f"[INFO] G1 geladen. Anzahl Gelenke: {robot.num_joints}")
    
    # Arm Indizes finden
    arm_indices = [
        i for i, n in enumerate(robot.joint_names) 
        if "right" in n and ("shoulder" in n or "elbow" in n)
    ]
    print(f"[INFO] Arm Gelenke: {arm_indices}")

    # Startposition merken
    default_pos = robot.data.default_joint_pos.clone()
    targets = default_pos.clone()
    
    sim_time = 0.0
    while simulation_app.is_running():
        sim_time += sim_cfg.dt
        
        # Winken
        if len(arm_indices) > 0:
            shoulder_idx = arm_indices[0]
            targets[:, shoulder_idx] = default_pos[:, shoulder_idx] + 1.0 * torch.sin(torch.tensor(sim_time * 2.0))
            
            if len(arm_indices) > 2:
                elbow_idx = arm_indices[2]
                targets[:, elbow_idx] = default_pos[:, elbow_idx] + 0.5 * torch.sin(torch.tensor(sim_time * 3.0))

        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        scene.update(dt=sim_cfg.dt)

if __name__ == "__main__":
    main()
    simulation_app.close()