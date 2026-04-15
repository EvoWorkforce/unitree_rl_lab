from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pose_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    termination_term_name: str = "goal_reached",
    success_threshold: float = 0.5,
) -> torch.Tensor:
    """Curriculum for pose command ranges.
    
    Progressively increases the position command ranges (pos_x, pos_y) based on
    the goal completion rate. When the robot successfully reaches goals in more
    than success_threshold of episodes, the difficulty increases.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to compute curriculum for.
        termination_term_name: Name of the goal_reached termination term.
        success_threshold: Fraction of successful episodes needed to increase difficulty (0.0-1.0).
    """
    command_term = env.command_manager.get_term("pose_command")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Check if the termination term exists
    if termination_term_name in env.termination_manager.active_terms:
        # Get the index for this termination term
        term_idx = env.termination_manager._term_name_to_term_idx[termination_term_name]
        
        # Get which environments terminated with goal_reached in the last episode
        # _last_episode_dones is a tensor of shape (num_envs, num_terms) with boolean values
        # Use ALL environments, not just env_ids, for a global success rate
        goal_reached_all = env.termination_manager._last_episode_dones[:, term_idx]
        
        # Calculate success rate as the fraction of ALL environments that reached the goal
        success_rate = torch.mean(goal_reached_all.float())
        
        # Increase difficulty if success rate exceeds threshold
        # Check periodically (every episode length)
        if env.common_step_counter % env.max_episode_length == 0:
            if success_rate > success_threshold:
                delta_command = torch.tensor([-0.1, 0.1], device=env.device)
                ranges.pos_x = torch.clamp(
                    torch.tensor(ranges.pos_x, device=env.device) + delta_command,
                    limit_ranges.pos_x[0],
                    limit_ranges.pos_x[1],
                ).tolist()
                ranges.pos_y = torch.clamp(
                    torch.tensor(ranges.pos_y, device=env.device) + delta_command,
                    limit_ranges.pos_y[0],
                    limit_ranges.pos_y[1],
                ).tolist()

    return torch.tensor(ranges.pos_x[1], device=env.device)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
