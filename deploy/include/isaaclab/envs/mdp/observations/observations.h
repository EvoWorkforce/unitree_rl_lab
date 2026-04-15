// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <spdlog/spdlog.h>

namespace isaaclab
{
    namespace mdp
    {

        REGISTER_OBSERVATION(base_ang_vel)
        {
            auto &asset = env->robot;
            auto &data = asset->data.root_ang_vel_b;
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(projected_gravity)
        {
            auto &asset = env->robot;
            auto &data = asset->data.projected_gravity_b;
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(joint_pos)
        {
            auto &asset = env->robot;
            std::vector<float> data;

            std::vector<int> joint_ids;
            try
            {
                joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
            }
            catch (const std::exception &e)
            {
            }

            if (joint_ids.empty())
            {
                data.resize(asset->data.joint_pos.size());
                for (size_t i = 0; i < asset->data.joint_pos.size(); ++i)
                {
                    data[i] = asset->data.joint_pos[i];
                }
            }
            else
            {
                data.resize(joint_ids.size());
                for (size_t i = 0; i < joint_ids.size(); ++i)
                {
                    data[i] = asset->data.joint_pos[joint_ids[i]];
                }
            }

            return data;
        }

        REGISTER_OBSERVATION(joint_pos_rel)
        {
            auto &asset = env->robot;
            std::vector<float> data;

            data.resize(asset->data.joint_pos.size());
            for (size_t i = 0; i < asset->data.joint_pos.size(); ++i)
            {
                data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
            }

            try
            {
                std::vector<int> joint_ids;
                joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
                if (!joint_ids.empty())
                {
                    std::vector<float> tmp_data;
                    tmp_data.resize(joint_ids.size());
                    for (size_t i = 0; i < joint_ids.size(); ++i)
                    {
                        tmp_data[i] = data[joint_ids[i]];
                    }
                    data = tmp_data;
                }
            }
            catch (const std::exception &e)
            {
            }

            return data;
        }

        REGISTER_OBSERVATION(joint_vel_rel)
        {
            auto &asset = env->robot;
            auto data = asset->data.joint_vel;

            try
            {
                const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

                if (!joint_ids.empty())
                {
                    data.resize(joint_ids.size());
                    for (size_t i = 0; i < joint_ids.size(); ++i)
                    {
                        data[i] = asset->data.joint_vel[joint_ids[i]];
                    }
                }
            }
            catch (const std::exception &e)
            {
            }
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(last_action)
        {
            auto data = env->action_manager->action();
            return std::vector<float>(data.data(), data.data() + data.size());
        };

        // Global override for velocity commands (used in hierarchical policies)
        // When set, this function returns velocity commands instead of reading from joystick
        inline std::function<std::vector<float>()> &velocity_commands_override()
        {
            static std::function<std::vector<float>()> instance = nullptr;
            return instance;
        }

        REGISTER_OBSERVATION(velocity_commands)
        {
            std::vector<float> obs(3);

            // Check for external velocity command override (e.g., from high-level navigation policy)
            if (velocity_commands_override())
            {
                return velocity_commands_override()();
            }

            auto &joystick = env->robot->data.joystick;

            const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

            obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
            obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
            obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

            return obs;
        }

        // Global override for pose command config (used in hierarchical policies)
        // When set, this returns the pose command ranges from the navigation config
        inline YAML::Node &pose_command_config_override()
        {
            static YAML::Node instance;
            return instance;
        }

        REGISTER_OBSERVATION(pose_command)
        {
            std::vector<float> obs(4);
            auto &goal_pose = env->robot->data.goal_pose;

            // Use override config if set (for hierarchical policies), otherwise use env config
            YAML::Node cfg;
            if (pose_command_config_override().IsDefined() && pose_command_config_override()["x"].IsDefined())
            {
                cfg = pose_command_config_override();
            }
            else
            {
                cfg = env->cfg["commands"]["pose_command"]["ranges"];
            }

            // Range keys differ across policies; support both naming conventions.
            // - Navigation policies: {x, y, yaw}
            // - Some pose- policies: {pos_x, pos_y, heading}
            auto x_range = cfg["x"].IsDefined() ? cfg["x"] : cfg["pos_x"];
            auto y_range = cfg["y"].IsDefined() ? cfg["y"] : cfg["pos_y"];
            auto yaw_range = cfg["yaw"].IsDefined() ? cfg["yaw"] : cfg["heading"];
            float min_safe_distance = cfg["min_safe_distance"].IsDefined() ? cfg["min_safe_distance"].as<float>() : 0.0f;
            float yaw_offset = cfg["yaw_offset"].IsDefined() ? cfg["yaw_offset"].as<float>() : 0.0f;

            // Return 4 values: [x, y, z, heading] to match Python training environment
            // pos_command_b (3D: x, y, z in base frame) + heading_command_b (1D: relative heading)
            obs[0] = std::clamp(goal_pose->x, x_range[0].as<float>(), x_range[1].as<float>());
            obs[1] = std::clamp(goal_pose->y, y_range[0].as<float>(), y_range[1].as<float>());
            obs[2] = 0.0f;  // z component (typically 0 for 2D navigation)
            obs[3] = std::clamp(goal_pose->theta + yaw_offset, yaw_range[0].as<float>(), yaw_range[1].as<float>());

            return obs;
        }

        REGISTER_OBSERVATION(gait_phase)
        {
            float period = params["period"].as<float>();
            float delta_phase = env->step_dt * (1.0f / period);

            env->global_phase += delta_phase;
            env->global_phase = std::fmod(env->global_phase, 1.0f);

            std::vector<float> obs(2);
            obs[0] = std::sin(env->global_phase * 2 * M_PI);
            obs[1] = std::cos(env->global_phase * 2 * M_PI);
            return obs;
        }

        REGISTER_OBSERVATION(zmp_xy)
        {
            std::vector<float> obs(1);
            obs[0] = 0.0f;
            return obs;
        }

        REGISTER_OBSERVATION(base_lin_vel)
        {
            // std::vector<float> obs(3);

            // // Estimate base linear velocity using change in goal pose over time
            // // Since the goal is fixed in world frame, its change in base frame reflects robot motion
            // static std::vector<float> prev_goal_pos_b = {0.0f, 0.0f, 0.0f};
            // static bool initialized = false;

            // auto &goal_pose = env->robot->data.goal_pose;

            // if (goal_pose && initialized)
            // {
            //     // Current goal position in base frame
            //     float curr_x = goal_pose->x;
            //     float curr_y = goal_pose->y;

            //     // Estimate velocity as negative change in goal position (robot moves opposite to goal motion in base frame)
            //     // velocity = -(current_pos - previous_pos) / dt
            //     float dt = env->step_dt;
            //     if (dt > 0.0f)
            //     {
            //         float vel_x = -(curr_x - prev_goal_pos_b[0]) / dt;
            //         float vel_y = -(curr_y - prev_goal_pos_b[1]) / dt;

            //         // Clamp to reasonable velocity range to handle goal resets or large updates
            //         // Typical humanoid velocities are within [-2, 2] m/s
            //         obs[0] = std::clamp(vel_x, -2.0f, 2.0f);
            //         obs[1] = std::clamp(vel_y, -2.0f, 2.0f);
            //         obs[2] = 0.0f; // z-velocity (vertical) is typically zero for legged locomotion
            //     }

            //     // Update previous goal position
            //     prev_goal_pos_b[0] = curr_x;
            //     prev_goal_pos_b[1] = curr_y;
            // }
            // else
            // {
            //     // Initialize on first call
            //     if (goal_pose)
            //     {
            //         prev_goal_pos_b[0] = goal_pose->x;
            //         prev_goal_pos_b[1] = goal_pose->y;
            //         initialized = true;
            //     }
            //     obs[0] = 0.0f;
            //     obs[1] = 0.0f;
            //     obs[2] = 0.0f;
            // }

            std::vector<float> obs(3);
            obs[0] = 0.0f;
            obs[1] = 0.0f;
            obs[2] = 0.0f;
            return obs;
        }

    }
}