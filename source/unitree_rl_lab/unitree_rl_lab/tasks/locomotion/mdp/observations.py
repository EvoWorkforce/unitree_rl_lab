from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def zmp_xy_l2_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Return L2 deviation of the dynamic ZMP from the midpoint of the feet
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # --- Dynamic ZMP calculation using CoM acceleration ---
    # Get CoM position, velocity
    com_xy = asset.data.root_com_pos_w[:, :2]  # (num_envs, 2)
    com_z = asset.data.root_com_pos_w[:, 2]    # (num_envs,)
    com_vel_xy = asset.data.root_com_vel_w[:, :2]  # (num_envs, 2)

    # Store previous CoM velocity in the environment (buffered per env)
    if not hasattr(env, "_prev_com_vel_xy_obs") or env._prev_com_vel_xy_obs is None or env._prev_com_vel_xy_obs.shape != com_vel_xy.shape:
        # Initialize buffer on first call or shape mismatch
        env._prev_com_vel_xy_obs = com_vel_xy.clone()

    # Compute acceleration (finite difference)
    com_acc_xy = (com_vel_xy - env._prev_com_vel_xy_obs) / env.step_dt  # (num_envs, 2)
    # Update buffer for next step
    env._prev_com_vel_xy_obs = com_vel_xy.clone()

    # Dynamic ZMP formula: zmp_xy = com_xy - com_z / g * com_acc_xy
    zmp_xy = com_xy - (com_z / 9.81).unsqueeze(-1) * com_acc_xy

    # Get feet positions and midpoint
    feet_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (num_envs, 2, 2)
    midpoint_xy = torch.mean(feet_xy, dim=1)  # (num_envs, 2)

    # Return L2 distance from ZMP to center of feet
    return torch.linalg.norm(zmp_xy - midpoint_xy, dim=1, keepdim=True)  # (num_envs, 1)