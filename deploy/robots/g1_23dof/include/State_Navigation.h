// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/hierarchical_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "goal_pose.h"
#include <yaml-cpp/yaml.h>

class State_Navigation : public FSMState
{
public:
    State_Navigation(int state_mode, std::string state_string);

    void enter()
    {
        // Ensure hierarchical overrides are enabled while this state is active.
        // (FSM states persist across transitions, so we must manage these globals explicitly.)
        if (pose_command_ranges.IsDefined())
        {
            isaaclab::mdp::pose_command_config_override() = pose_command_ranges;
        }

        // set gain
        for (int i = 0; i < env->low_level_env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->low_level_env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->low_level_env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]
                                    {
            using clock = std::chrono::high_resolution_clock;
            // Use low-level step_dt for the control loop
            const std::chrono::duration<double> desiredDuration(env->low_level_env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            } });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }

        // Disable global overrides so other FSM states (e.g., PoseTracking) read their own env configs.
        isaaclab::mdp::velocity_commands_override() = nullptr;
        isaaclab::mdp::pose_command_config_override().reset();
    }

private:
    std::unique_ptr<isaaclab::HierarchicalRLEnv> env;
    isaaclab::GoalPoseSubscription::SharedPtr goal_pose_sub;

    // Cached pose command ranges from the navigation policy config.
    YAML::Node pose_command_ranges;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Navigation)
