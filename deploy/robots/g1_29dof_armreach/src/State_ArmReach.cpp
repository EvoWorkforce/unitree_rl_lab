#include "FSM/State_ArmReach.h"

#include <spdlog/spdlog.h>

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

    // cfg is a YAML::Node of type map
    static auto cfg = env->cfg["commands"]["pose_command"]["ranges"];

    static std::vector<float> current_target = {0.3f, 0.0f, 0.4f, 0.0f,
                                                0.0f, 0.0f, 0.0f};

    
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
    //vector with the id's of the joints of the left arm
    std::vector<int> action_joints{11, 15, 19, 21, 23, 25, 27};
    auto action = env->action_manager->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        
        // Check if "i" is one of the joints of the arm
        int ctn = std::count(action_joints.begin(), action_joints.end(), i);

        if(ctn <= 0)
        {
            // Change mode to 0 if count is 0 (not included in the vector)
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].mode() = 0;
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kp() = 0;
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kd() = 0;
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].dq() = 0;
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].tau() = 0;
            
        }   else 
        {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() =
                action[i];
        }

        // spdlog::info("I = {}", i);
    }
}