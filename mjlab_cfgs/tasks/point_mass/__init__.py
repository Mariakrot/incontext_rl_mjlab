# src/mjlab/tasks/point_mass/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner
from .env_cfg import point_mass_env_cfg
from .rl_cfg import point_mass_ppo_cfg

register_mjlab_task(
    task_id="Mjlab-PointMass-Reach",
    env_cfg=point_mass_env_cfg(),
    play_env_cfg=point_mass_env_cfg(play=True),
    rl_cfg=point_mass_ppo_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)