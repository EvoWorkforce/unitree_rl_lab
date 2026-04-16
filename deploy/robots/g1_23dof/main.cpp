#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_Navigation.h"
#include "State_NavigationPID.h"
#include "State_PoseTracking.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;
std::shared_ptr<isaaclab::KeyboardInput> FSMState::keyboard_input = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char **argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Read ROS_DOMAIN_ID from environment variable, default to 0 if not set
    int ros_domain_id = 0;
    const char *env_p = std::getenv("ROS_DOMAIN_ID");
    if (env_p != nullptr)    {
        ros_domain_id = std::stoi(std::string(env_p));
    }
    spdlog::info("Using ROS_DOMAIN_ID: {}", ros_domain_id);

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(ros_domain_id, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if (!FSMState::lowcmd->check_mode_machine(FSMState::lowstate))
    {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize keyboard input if enabled in config
    bool enable_keyboard = false;
    try
    {
        enable_keyboard = param::config["FSM"]["enable_keyboard"].as<bool>();
    }
    catch (...)
    {
    }

    if (enable_keyboard)
    {
        spdlog::info("Keyboard input enabled for FSM transitions");
        FSMState::keyboard_input = std::make_shared<isaaclab::KeyboardInput>();
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";
    if (enable_keyboard)
    {
        std::cout << "Keyboard input is enabled. Use key_X transitions in config.\n";
    }

    while (true)
    {
        sleep(1);
    }

    return 0;
}
