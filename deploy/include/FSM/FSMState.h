#pragma once

#include "FSM/BaseState.h"
#ifdef UNITREE_USE_JOYSTICK_INJECTOR
#include "JoystickInjector.h"
#endif
#include "Types.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "keyboard_input.h"
#include "param.h"
#include "unitree_joystick_dsl_extended.hpp"
// #warning "FSMState.h WITH JOYSTICK INJECTOR IS BEING USED"

class FSMState : public BaseState
{
public:
  FSMState(int state, std::string state_string)
      : BaseState(state, state_string)
  {
    spdlog::info("Initializing State_{} ...", state_string);

    auto transitions = param::config["FSM"][state_string]["transitions"];

    if (transitions)
    {
      auto transition_map =
          transitions.as<std::map<std::string, std::string>>();

      for (auto it = transition_map.begin(); it != transition_map.end(); ++it)
      {
        std::string target_fsm = it->first;
        if (!FSMStringMap.right.count(target_fsm))
        {
          spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
          continue;
        }

        int fsm_id = FSMStringMap.right.at(target_fsm);

        std::string condition = it->second;

        // spdlog::info("FSM transition: {} -> {} when [{}]", state_string,
        //              target_fsm, condition);

        // Use extended DSL that supports both joystick and keyboard
        auto func = unitree::common::dsl::CompileExpressionExtended(condition);
        registered_checks.emplace_back(std::make_pair(
            [func]() -> bool
            {
              unitree::common::dsl::InputContext ctx;
              ctx.joystick = &FSMState::lowstate->joystick;
              ctx.keyboard = FSMState::keyboard_input.get();
              return func(ctx);
            },
            fsm_id));
      }
    }

    // register for all states
    registered_checks.emplace_back(
        std::make_pair([]() -> bool
                       { return lowstate->isTimeout(); },
                       FSMStringMap.right.at("Passive")));
  }

  void pre_run()
  {
    lowstate->update();

    // static int _p = 0;
    // if ((_p++ % 1000) == 0) {
    //   std::cout << "[DBG] &lowstate=" << lowstate.get()
    //             << " &lowstate->joystick=" << (void *)&lowstate->joystick
    //             << "\n"
    //             << std::flush;
    // }

    static double acc = 0.0;
    acc += 0.001; // dt is 0.001 in CtrlFSM
    if (acc >= 1.0)
    {
      acc = 0.0;
      auto &j = FSMState::lowstate->joystick;
      // spdlog::info(
      //     "DBG joy_ptr={} LT pressed={} on_pressed={} pressed_time={:.3f}",
      //     (void *)&FSMState::lowstate->joystick,
      //     FSMState::lowstate->joystick.LT.pressed,
      //     FSMState::lowstate->joystick.LT.on_pressed,
      //     FSMState::lowstate->joystick.LT.pressed_time);
    }

#ifdef UNITREE_USE_JOYSTICK_INJECTOR
    JoystickInjector::Instance().start();
    JoystickInjector::Instance().apply(lowstate->joystick);
#endif
    if (keyboard)
      keyboard->update();
    if (keyboard_input)
      keyboard_input->update();
  }

  void post_run() { lowcmd->unlockAndPublish(); }

  static std::unique_ptr<LowCmd_t> lowcmd;
  static std::shared_ptr<LowState_t> lowstate;
  static std::shared_ptr<Keyboard> keyboard;
  static std::shared_ptr<isaaclab::KeyboardInput> keyboard_input;
};
