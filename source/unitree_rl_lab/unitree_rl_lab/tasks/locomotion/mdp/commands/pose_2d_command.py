from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformPose2dCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelPose2dCommandCfg(UniformPose2dCommandCfg):
    limit_ranges: UniformPose2dCommandCfg.Ranges = MISSING
