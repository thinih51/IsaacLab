import gymnasium as gym
import torch
import math
import os
from datetime import datetime
import argparse

# IsaacLab Imports
from isaaclab.app import AppLauncher

# Argumente parsen
parser = argparse.ArgumentParser(description="Train G1 to Walk")
parser.add_argument("--num_envs", type=int, default=4096, help="Anzahl der Roboter (Standard 4096)")
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
            activate_contact_sensors=True, # Wichtig für Crash-Erkennung
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.85)),
        
        # WICHTIG: "fix_root_link" IST HIER ENTFERNT WORDEN!
        # Die Freiheit kommt aus der USD-Datei selbst (Moveable Base).
        
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
    
    # Domain Randomization (Macht den Roboter robust für langes Training)
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )

@configclass
class CommandsCfg:
    """Ziele: Vorwärts laufen."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.5), # Schneller laufen lernen
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class RewardsCfg:
    """Belohnungen."""
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"std": math.sqrt(0.25), "command_name": "base_velocity"}
    )
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.0005)
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

# --- 5. TRAINING ---
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

def main():
    env_cfg = G1EnvCfg()
    env_base = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlWrapper(env_base)

    log_dir = os.path.join("logs", "rsl_rl", "g1_walking", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
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
            "max_iterations": 10000, # LANGES TRAINING FÜR DIE NACHT
            "save_interval": 200,
        },
        log_dir=log_dir,
        device="cuda:0"
    )

    print(f"[INFO] Start Training... Logs in {log_dir}")
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True)
    
    print("[INFO] Training fertig!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()