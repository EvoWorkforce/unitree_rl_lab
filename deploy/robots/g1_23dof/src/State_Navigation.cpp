#include "State_Navigation.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_Navigation::State_Navigation(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto navigation_policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // Get velocity policy directory (required for hierarchical deployment)
    std::string velocity_policy_path = "config/policy/velocity/v2.6"; // default
    try
    {
        velocity_policy_path = cfg["velocity_policy_dir"].as<std::string>();
    }
    catch (const std::exception &e)
    {
        // Use default
    }
    auto velocity_policy_dir = param::parser_policy_dir(velocity_policy_path);

    // Get decimation ratio (how many low-level steps per high-level step)
    int high_level_decimation = 10; // default from training config
    try
    {
        high_level_decimation = cfg["high_level_decimation"].as<int>();
    }
    catch (const std::exception &e)
    {
        // Use default
    }

    // Create goal pose subscription
    std::string goal_pose_topic = "rt/goal_pose";
    try
    {
        goal_pose_topic = cfg["goal_pose_topic"].as<std::string>();
    }
    catch (const std::exception &e)
    {
        // Use default topic
    }
    goal_pose_sub = std::make_shared<isaaclab::GoalPoseSubscription>(goal_pose_topic);

    // Create articulation with goal pose subscription
    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
        FSMState::lowstate, goal_pose_sub);

    // Load configs for both policies
    auto navigation_cfg = YAML::LoadFile(navigation_policy_dir / "params" / "deploy.yaml");
    auto velocity_cfg = YAML::LoadFile(velocity_policy_dir / "params" / "deploy.yaml");

    // Cache pose command ranges so we can re-enable overrides on future re-entries.
    // (FSM states persist, so the HierarchicalRLEnv constructor won't rerun.)
    try
    {
        pose_command_ranges = navigation_cfg["commands"]["pose_command"]["ranges"];
    }
    catch (const std::exception &e)
    {
        // Leave undefined
    }

    // Create hierarchical environment
    env = std::make_unique<isaaclab::HierarchicalRLEnv>(
        navigation_cfg,
        velocity_cfg,
        articulation,
        high_level_decimation);

    // Load policies
    env->high_level_alg = std::make_unique<isaaclab::OrtRunner>(navigation_policy_dir / "exported" / "policy.onnx");
    env->low_level_env->alg = std::make_unique<isaaclab::OrtRunner>(velocity_policy_dir / "exported" / "policy.onnx");

    // Enable debug prints (goal pose and velocity commands)
    try
    {
        env->debug_print = cfg["debug_print"].as<bool>();
    }
    catch (const std::exception &e)
    {
        env->debug_print = true; // Default to true for debugging
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            { return isaaclab::mdp::bad_orientation(env->low_level_env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
}

void State_Navigation::run()
{
    auto action = env->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
