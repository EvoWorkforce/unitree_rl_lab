// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include <functional>

namespace isaaclab
{

    /**
     * @brief Hierarchical RL Environment for chaining a high-level policy with a low-level policy.
     *
     * This class manages two policies:
     * 1. High-level policy (e.g., navigation): takes high-level observations → outputs velocity commands
     * 2. Low-level policy (e.g., velocity tracking): takes velocity commands + robot state → outputs joint actions
     *
     * The high-level policy runs at a lower frequency (high_level_decimation * low_level step_dt).
     */
    class HierarchicalRLEnv
    {
    public:
        /**
         * @brief Construct a hierarchical environment.
         *
         * @param high_level_cfg YAML config for high-level policy (navigation)
         * @param low_level_cfg YAML config for low-level policy (velocity)
         * @param robot Shared articulation robot
         * @param high_level_decimation How many low-level steps per high-level step
         */
        HierarchicalRLEnv(
            YAML::Node high_level_cfg,
            YAML::Node low_level_cfg,
            std::shared_ptr<Articulation> robot,
            int high_level_decimation = 10)
            : robot(robot), high_level_decimation(high_level_decimation)
        {
            // The low-level environment runs the velocity policy
            low_level_env = std::make_unique<ManagerBasedRLEnv>(low_level_cfg, robot);

            // High-level step_dt is the navigation policy's step_dt
            step_dt = high_level_cfg["step_dt"].as<float>();

            // Set pose command config override for high-level observations
            if (high_level_cfg["commands"]["pose_command"]["ranges"].IsDefined())
            {
                mdp::pose_command_config_override() = high_level_cfg["commands"]["pose_command"]["ranges"];
            }

            // Create high-level observation manager only (no action manager needed as output is velocity commands)
            high_level_obs_manager = std::make_unique<ObservationManager>(high_level_cfg["observations"], low_level_env.get());

            // Store velocity command ranges for clamping
            if (low_level_cfg["commands"]["base_velocity"]["ranges"].IsDefined())
            {
                auto ranges = low_level_cfg["commands"]["base_velocity"]["ranges"];
                vel_cmd_ranges[0] = {ranges["lin_vel_x"][0].as<float>(), ranges["lin_vel_x"][1].as<float>()};
                vel_cmd_ranges[1] = {ranges["lin_vel_y"][0].as<float>(), ranges["lin_vel_y"][1].as<float>()};
                vel_cmd_ranges[2] = {ranges["ang_vel_z"][0].as<float>(), ranges["ang_vel_z"][1].as<float>()};
            }
        }

        void reset()
        {
            low_level_counter = 0;
            high_level_action.assign(3, 0.0f); // velocity commands: [vx, vy, wz]

            // Set up velocity commands override to use high-level policy output
            mdp::velocity_commands_override() = [this]() -> std::vector<float>
            {
                return this->high_level_action;
            };

            low_level_env->reset();
            high_level_obs_manager->reset();
        }

        /**
         * @brief Step the hierarchical environment.
         *
         * Runs the high-level policy every high_level_decimation steps,
         * and the low-level policy every step.
         */
        void step()
        {
            robot->update();
            low_level_env->episode_length += 1;

            // Run high-level policy at lower frequency
            if (low_level_counter % high_level_decimation == 0)
            {
                auto high_level_obs = high_level_obs_manager->compute();
                high_level_action = high_level_alg->act(high_level_obs);

                // Clamp velocity commands to valid ranges
                for (int i = 0; i < 3; ++i)
                {
                    high_level_action[i] = std::clamp(high_level_action[i], vel_cmd_ranges[i][0], vel_cmd_ranges[i][1]);
                }

                // Debug print
                if (debug_print)
                {
                    auto &goal_pose = robot->data.goal_pose;
                    std::cout << "[Navigation] Goal Pose: x=" << goal_pose->x
                              << ", y=" << goal_pose->y
                              << ", theta=" << goal_pose->theta << std::endl;
                    std::cout << "[Navigation] Velocity Cmd: vx=" << high_level_action[0]
                              << ", vy=" << high_level_action[1]
                              << ", wz=" << high_level_action[2] << std::endl;
                }
            }

            low_level_counter++;

            // Run low-level policy with velocity commands from high-level policy
            // The velocity_commands_override is already set to return high_level_action
            auto low_level_obs = low_level_env->observation_manager->compute();
            auto low_level_action = low_level_env->alg->act(low_level_obs);
            low_level_env->action_manager->process_action(low_level_action);
        }

        /**
         * @brief Cleanup when environment is destroyed.
         */
        ~HierarchicalRLEnv()
        {
            // Clear the velocity commands override
            mdp::velocity_commands_override() = nullptr;
            // Clear the pose command config override
            mdp::pose_command_config_override().reset();
        }

        /**
         * @brief Get the velocity commands from the high-level policy.
         * Used to inject into low-level observation "velocity_commands".
         */
        const std::vector<float> &get_velocity_commands() const
        {
            return high_level_action;
        }

        /**
         * @brief Get the processed joint actions from the low-level policy.
         */
        std::vector<float> processed_actions()
        {
            return low_level_env->action_manager->processed_actions();
        }

        // Public members
        float step_dt; // High-level step dt (navigation policy)
        std::shared_ptr<Articulation> robot;
        std::unique_ptr<ManagerBasedRLEnv> low_level_env;
        std::unique_ptr<Algorithms> high_level_alg; // Navigation policy
        bool debug_print = false;                   // Enable debug prints for goal pose and velocity commands

    private:
        std::unique_ptr<ObservationManager> high_level_obs_manager;
        std::vector<float> high_level_action{0.0f, 0.0f, 0.0f}; // Velocity commands
        int high_level_decimation;
        int low_level_counter = 0;
        std::array<std::array<float, 2>, 3> vel_cmd_ranges{{{-1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, 1.0f}}};
    };

}; // namespace isaaclab
