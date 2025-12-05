import gymnasium as gym
import torch
import math
import os
import glob
from datetime import datetime
import argparse

# IsaacLab Imports
from isaaclab.app import AppLauncher

# Argumente
parser = argparse.ArgumentParser(description="Play G1 Walking Policy")
parser.add_argument("--num_envs", type=int, default=16, help="Anzahl der Roboter zum Zuschauen")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# App starten
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports nach App-Start
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp 
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

# --- PFAD ANPASSEN ---
USD_PATH = "C:/Users/tshed/Documents/WIP/IsaacLab/resources/g1/g1_29dof/g1_29dof.usd"
# ---------------------

# --- 1. SZENEN KONFIGURATION ---
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Die 3D-Welt: Boden, Licht und Roboter."""
    
    # A. Boden
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # B. Licht
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # C. Der Roboter
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=True, 
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.82)),
        
        # FEHLER BEHOBEN: fix_root_link entfernt (das ist jetzt Standard bzw. im USD)
        
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=200.0, damping=5.0,
            ),
        },
    )

    # D. Kontakt-Sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3,
        track_air_time=True,
    )

# --- 2. MANAGER KONFIGURATIONEN ---

@configclass
class ActionsCfg:
    """Wir steuern alle Gelenke."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class EventsCfg:
    """Reset und Randomisierung."""
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        },
    )
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

@configclass
class CommandsCfg:
    """Ziele: Vorwärts laufen."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0), 
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(-0.5, 0.5),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class RewardsCfg:
    """Belohnungen (Dummy für Play)."""
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)

@configclass
class TerminationsCfg:
    """Game Over Bedingungen."""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        }
    )

@configclass
class ObservationsCfg:
    """Was sieht das Netz?"""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)
        commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    policy = PolicyCfg()

# --- 3. HAUPT UMGEBUNG ---

@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    """Verknüpfung aller Teile."""
    
    scene = G1SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=True)
    
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0

# --- 4. DER WRAPPER ---
class TensorDict(dict):
    def to(self, device): return self

class RslRlWrapper:
    def __init__(self, env):
        self.env = env
    
    def __getattr__(self, attr):
        return getattr(self.env, attr)

    @property
    def num_envs(self): return self.env.num_envs
    
    @property
    def num_obs(self): return self.env.observation_manager.group_obs_dim["policy"][0]

    @property
    def num_privileged_obs(self): return None 

    @property
    def num_actions(self): return self.env.action_manager.total_action_dim

    @property
    def max_episode_length(self): return self.env.max_episode_length_s

    @property
    def device(self): return self.env.device

    def get_observations(self):
        return TensorDict({"policy": self.env.observation_manager.compute_group("policy")})

    def reset(self):
        obs, _ = self.env.reset()
        return TensorDict(obs)

    def step(self, actions):
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        return TensorDict(obs), rew, terminated | truncated, extras
    
    def close(self):
        self.env.close()

# --- 5. PLAY LOOP ---
def main():
    # Environment erstellen
    env_cfg = G1EnvCfg()
    env_base = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlWrapper(env_base)

    # Log Ordner finden
    log_root = os.path.join("logs", "rsl_rl", "g1_walking")
    if not os.path.exists(log_root):
        print(f"[ERROR] Kein Log-Ordner gefunden unter {log_root}!")
        return

    # Neuesten Ordner nehmen
    all_folders = glob.glob(os.path.join(log_root, "*"))
    latest_log = max(all_folders, key=os.path.getctime)
    print(f"[INFO] Lade Training aus: {latest_log}")

    # Runner laden (Konfig muss mit Train übereinstimmen!)
    runner = OnPolicyRunner(
        env=env,
        train_cfg={
            "seed": 42,
            "runner_class_name": "OnPolicyRunner",
            "device": "cuda:0",
            "obs_groups": {"policy": ["policy"], "critic": ["policy"]}, 
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 1.0,
                "actor_hidden_dims": [256, 160, 128],
                "critic_hidden_dims": [256, 160, 128],
                "activation": "elu",
            },
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 1.0e-3,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
            "num_steps_per_env": 24,
            "max_iterations": 500,
            "save_interval": 50,
        },
        log_dir=latest_log,
        device="cuda:0"
    )
    
    # Modell laden
    # Wir suchen nach dem neuesten Modell (egal ob 500, 1000 oder abgebrochen)
    models = glob.glob(os.path.join(latest_log, "model_*.pt"))
    if not models:
        print("[ERROR] Kein model_*.pt gefunden! Training war wohl zu kurz.")
        return

    model_path = max(models, key=os.path.getctime)
    print(f"[INFO] Lade Modell: {model_path}")
    runner.load(model_path)
    
    policy = runner.get_inference_policy(device="cuda:0")
    obs = env.get_observations()
    
    print("[INFO] Starte Playback...")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # Aktionen berechnen
            actions = policy(obs) # Hier übergeben wir das ganze Dictionary
            
        # Schritt ausführen
        obs, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()