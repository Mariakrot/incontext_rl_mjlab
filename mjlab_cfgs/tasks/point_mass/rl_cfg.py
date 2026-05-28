# src/mjlab/tasks/point_mass/rl_cfg.py


def point_mass_ppo_cfg() -> dict:
    return {
        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"]
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 64],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,        # Начальный уровень исследования (шума)
                "std_type": "scalar"    # Тип дисперсии (скалярный шум)
            }
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 64],
            "activation": "elu",
            "obs_normalization": True
        },
        "algorithm": {
            "class_name": "PPO", 
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0
        },
        "experiment_name": "point_mass",
        "save_interval": 50,
        "num_steps_per_env": 24,
        "max_iterations": 800,
        "upload_model": False
    }