# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(env: ManagerBasedRLEnv, command_name: str, distance_threshold: float = 0.2, heading_threshold: float = 0.1) -> torch.Tensor:
    """Terminate episode when robot reaches the goal position.
    
    Args:
        env: The environment instance.
        command_name: The name of the command term.
        distance_threshold: Distance threshold in meters to consider goal reached.
        heading_threshold: Heading threshold in radians to consider goal reached.
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    command = env.command_manager.get_command(command_name)
    pos_error = command[:, :2]  # x, y position error in base frame
    distance = torch.norm(pos_error, dim=1)
    heading_error = command[:, 3].abs()  # heading error in radians
    return (distance < distance_threshold) & (heading_error < heading_threshold)
