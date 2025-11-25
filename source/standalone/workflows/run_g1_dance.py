import argparse
import torch
from isaaclab.app import AppLauncher

# Argumente parsen
parser = argparse.ArgumentParser(description="Unitree G1 Dance Party")
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

# ---------------------------------------------------------
# PFAD ANPASSEN (Falls nötig)
USD_PATH = "C:/Users/tshed/Documents/WIP/IsaacLab/resources/g1/g1_29dof/g1_29dof.usd"
# ---------------------------------------------------------

@configclass
class UnitreeG1Cfg(ArticulationCfg):
    """Konfiguration für den G1 Roboter (Tanz-Modus)."""
    
    # 1. Adresse im Szenengraph
    prim_path = "/World/Robot"
    
    # 2. USD Datei laden
    spawn = sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
    )
    
    # 3. Start-Position: Etwas höher (1.1m), damit die Beine Platz haben
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
    )
    
    # 4. Basis fixieren (WICHTIG: Sonst fällt er um)
    fix_root_link = True
    
    # 5. Aktuatoren: Etwas steifer für schnelle Bewegungen
    actuators = {
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=300.0,
            damping=20.0,
        ),
    }

def main():
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Roboter hinzufügen
    robot_cfg = UnitreeG1Cfg()
    robot = Articulation(cfg=robot_cfg)
    scene.articulations["g1"] = robot

    # Start
    sim.reset()
    print(f"[INFO] G1 bereit zum Tanzen! Gelenke: {robot.num_joints}")

    # --- Gelenk-Indizes suchen ---
    # Wir suchen die passenden Nummern für Arme, Beine und Hüfte
    
    # Schultern (Pitch = vor/zurück schwingen)
    l_shoulder_indices = [i for i, n in enumerate(robot.joint_names) if "left" in n and "shoulder_pitch" in n]
    r_shoulder_indices = [i for i, n in enumerate(robot.joint_names) if "right" in n and "shoulder_pitch" in n]
    
    # Beine/Hüfte (Pitch = Gehen)
    l_hip_indices = [i for i, n in enumerate(robot.joint_names) if "left" in n and "hip_pitch" in n]
    r_hip_indices = [i for i, n in enumerate(robot.joint_names) if "right" in n and "hip_pitch" in n]

    # Taille (Yaw = Drehen)
    waist_indices = [i for i, n in enumerate(robot.joint_names) if "waist_yaw" in n]

    print(f"[INFO] L-Arm: {l_shoulder_indices}, R-Arm: {r_shoulder_indices}")
    
    # Startposition merken
    default_pos = robot.data.default_joint_pos.clone()
    targets = default_pos.clone()
    
    sim_time = 0.0
    
    while simulation_app.is_running():
        sim_time += sim_cfg.dt
        
        # --- Die Tanz-Logik ---
        
        # Wir erzeugen zwei Wellen: eine schnelle und eine langsame
        fast_wave = torch.sin(torch.tensor(sim_time * 6.0))  # Marschier-Tempo
        slow_wave = torch.sin(torch.tensor(sim_time * 3.0))  # Dreh-Tempo

        # 1. Arme schwingen (Gegenläufig: Links vor, Rechts zurück)
        if l_shoulder_indices:
            targets[:, l_shoulder_indices[0]] = default_pos[:, l_shoulder_indices[0]] + 0.8 * fast_wave
        if r_shoulder_indices:
            targets[:, r_shoulder_indices[0]] = default_pos[:, r_shoulder_indices[0]] - 0.8 * fast_wave

        # 2. Beine bewegen (Gegenläufig zu den Armen = wie beim Gehen)
        if l_hip_indices:
            targets[:, l_hip_indices[0]] = default_pos[:, l_hip_indices[0]] - 0.6 * fast_wave
        if r_hip_indices:
            targets[:, r_hip_indices[0]] = default_pos[:, r_hip_indices[0]] + 0.6 * fast_wave

        # 3. Oberkörper drehen (Twist)
        if waist_indices:
            targets[:, waist_indices[0]] = 0.3 * slow_wave

        # Befehl anwenden
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        
        sim.step()
        scene.update(dt=sim_cfg.dt)

if __name__ == "__main__":
    main()
    simulation_app.close()