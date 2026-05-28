# src/mjlab/tasks/point_mass/env_cfg.py
"""Point mass task: reach target position."""
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_cfgs.assets.zoo.robots.point_mass.point_mass_constants import get_point_mass_robot_cfg
from mjlab.envs import mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

def point_mass_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # --- Scene ---
    scene_cfg = SceneCfg(
        num_envs=256 if not play else 32,
        extent=1.0,
        entities={"robot": get_point_mass_robot_cfg()},
    )
    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        distance=2.0, elevation=30.0, azimuth=45.0,
    )
    sim_cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.02, iterations=1))

    # --- Actions: force control in x/y ---
    actions = {
        "force": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=("free_x", "free_y"),
            scale=1.0,  # max force = 1N
        ),
    }

    # --- Observations: [x, y, vx, vy, target_x, target_y] ---
    def get_target_offset(env):
        # Fixed target at (1.0, 1.0) for simplicity
        return torch.tensor([[1.0, 1.0]] * env.num_envs, device=env.device)
    
    obs_terms = {
        "pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, :2] / 2.0),
        "vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, :2] / 2.0),
        "target": ObservationTermCfg(func=get_target_offset),
    }
    observations = {
        "actor": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True, enable_corruption=False),
    }

    # --- Rewards ---
    def dist_to_target(env):
        pos = env.sim.data.qpos[:, :2]
        target = torch.tensor([[1.0, 1.0]], device=env.device)
        return -torch.norm(pos - target, dim=1)  # negative distance
    
    def effort_penalty(env):
        return -0.001 * torch.sum(env.sim.data.ctrl ** 2, dim=1)
    
    rewards = {
        "reach": RewardTermCfg(func=dist_to_target, weight=10.0),
        "effort": RewardTermCfg(func=effort_penalty, weight=1.0),
    }

    # --- Events: random reset ---
    events = {
        "reset": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(name="robot"),  # ← ЯВНЫЙ ОБЪЕКТ, НЕ DICT
                "position_range": (-0.5, 0.5),
                "velocity_range": (-0.2, 0.2),
            },
        ),
    }

    # --- Terminations ---
    terminations = {
        "timeout": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "success": TerminationTermCfg(
            func=lambda env: torch.norm(env.sim.data.qpos[:, :2] - torch.tensor([[1.0, 1.0]], device=env.device), dim=1) < 0.1,
            time_out=False,
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        events=events,
        terminations=terminations,
        sim=sim_cfg,
        viewer=viewer_cfg,
        decimation=1,
        episode_length_s=10.0,
    )