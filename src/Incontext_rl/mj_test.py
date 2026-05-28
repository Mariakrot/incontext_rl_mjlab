# src/incontext/mj_test.py
import sys
import os
import argparse
import torch
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mjlab_cfgs.tasks.point_mass.env_cfg import point_mass_env_cfg
from mjlab_cfgs.tasks.point_mass.rl_cfg import point_mass_ppo_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="./logs/point_mass")
    args = parser.parse_args()

    env_cfg = point_mass_env_cfg()
    rl_cfg = point_mass_ppo_cfg()

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.device = args.device
    rl_cfg["max_iterations"] = args.max_iters

    os.makedirs(args.log_dir, exist_ok=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env=RslRlVecEnvWrapper(env)
    runner = MjlabOnPolicyRunner(env=env, train_cfg=rl_cfg, log_dir=args.log_dir, device=args.device)
    print(f" Start | {args.num_envs} envs | {args.device} | {args.max_iters} iters")

    # Этот метод сам соберет траектории, обновит нейросеть, запишет логи в TensorBoard и сохранит модели
    runner.learn(num_learning_iterations=rl_cfg["max_iterations"], init_at_random_ep_len=True)

    # Финальное сохранение (если нужно сделать именно в конце скрипта)
    final_path = os.path.join(args.log_dir, "model_final.pt")
    runner.save(final_path)
    
    env.close()
    print(f"Done. Logs: {os.path.abspath(args.log_dir)}")
    print("Для просмотра графиков запустите: tensorboard --logdir ./logs")

if __name__ == "__main__":
    main()