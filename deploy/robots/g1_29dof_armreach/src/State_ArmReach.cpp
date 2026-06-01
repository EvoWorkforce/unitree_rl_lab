#include "FSM/State_ArmReach.h"

#include <unordered_map>
#include <vector>

#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "unitree_articulation.h"

namespace isaaclab
{
REGISTER_OBSERVATION(pose_command)
{
    // std::string key = FSMState::keyboard->key();
    // static auto cfg = env->cfg["commands"]["pose_command"]["ranges"];

    static std::vector<float> current_target = {0.3f, 0.0f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f};
    return current_target;
}

}  // namespace isaaclab

State_ArmReach::State_ArmReach(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir =
        param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate));
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" /
                                                     "policy.onnx");

    // this->registered_checks.emplace_back(
    //     std::make_pair(
    //         [&]()->bool{ return
    //         isaaclab::mdp::bad_orientation(env.get(), 1.0); },
    //         FSMStringMap.right.at("Passive")
    //     )
    // );
}

void State_ArmReach::run()
{
    auto action = env->action_manager->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() =
            action[i];
    }
}