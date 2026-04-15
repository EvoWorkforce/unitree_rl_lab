# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def print_reset(env: ManagerBasedRLEnv, env_ids: Sequence[int]):
    """Print when robot env resets"""
    print(f"=== Episode Reset ===")


def print_pose_error(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "pose_command"):
    """Print the current pose error (observation) for debugging during episode.
    
    This function prints the pose error in the robot's base frame, showing how far
    the robot needs to move to reach the goal.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs (not used, but required for interval events).
        command_name: The name of the command term to print.
    """
    # Get the command observation for the first environment (env 0)
    command = env.command_manager.get_command(command_name)
    
    # Get the command term to access internal state
    command_term = env.command_manager.get_term(command_name)
    
    # Get robot data
    robot = env.scene[command_term.cfg.asset_name]
    
    # Print only for the first environment to avoid spam
    if env.num_envs > 0:
        # Observation (base frame errors)
        pos_x_b = command[0, 0].item()
        pos_y_b = command[0, 1].item()
        heading_b = command[0, 3].item()  # Index 3 is heading, not index 2 (which is z)
        distance = (pos_x_b**2 + pos_y_b**2)**0.5
        
        # World frame commands and robot state
        pos_x_w = command_term.pos_command_w[0, 0].item()
        pos_y_w = command_term.pos_command_w[0, 1].item()
        heading_w_cmd = command_term.heading_command_w[0].item()
        heading_w_robot = robot.data.heading_w[0].item()
        
        # Metrics
        error_pos_2d = command_term.metrics["error_pos_2d"][0].item()
        error_heading = command_term.metrics["error_heading"][0].item()
        
        print(f"[Pose Error] Env 0:")
        print(f"  Obs (base frame):  x={pos_x_b:.3f}m, y={pos_y_b:.3f}m, heading_err={heading_b:.3f}rad, dist={distance:.3f}m")
        print(f"  Goal (world frame): x={pos_x_w:.3f}m, y={pos_y_w:.3f}m, heading={heading_w_cmd:.3f}rad")
        print(f"  Robot heading_w:    {heading_w_robot:.3f}rad")
        print(f"  Metrics:            pos_err={error_pos_2d:.3f}m, heading_err={error_heading:.3f}rad")
