# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable_wandb", action="store_true", default=False, help="Disable WandB logging (enabled by default)."
)
parser.add_argument(
    "--run_path",
    type=str,
    default=None,
    help="WandB run path (format: entity/project/run_id) to download checkpoint from and upload results to.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate WandB arguments before launching IsaacSim
if not args_cli.disable_wandb and not args_cli.run_path:
    print("[ERROR] WandB is enabled but no --run_path specified.")
    print("[ERROR] Please provide a WandB run path using --run_path entity/project/run_id")
    print("[ERROR] Or disable WandB with --disable_wandb")
    exit(1)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import wandb

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _download_latest_wandb_checkpoint(run_path: str, download_dir: str) -> str | None:
    """Download the latest model_*.pt checkpoint and params/deploy.yaml from a WandB run's files."""

    api = wandb.Api()
    run = api.run(run_path)

    checkpoint_files = []
    deploy_yaml_file = None

    for f in run.files():
        name = os.path.basename(f.name)
        if name.startswith("model_") and name.endswith(".pt"):
            try:
                step = int(name[len("model_") : -len(".pt")])
                checkpoint_files.append((step, f))
            except ValueError:
                pass
        elif f.name == "params/deploy.yaml":
            deploy_yaml_file = f

    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: x[0])
    latest_step, latest_file = checkpoint_files[-1]

    os.makedirs(download_dir, exist_ok=True)

    # Download checkpoint
    downloaded_file = latest_file.download(root=download_dir, replace=True)
    checkpoint_path = downloaded_file.name
    downloaded_file.close()

    print(f"[INFO] Downloaded WandB checkpoint step {latest_step} to: {checkpoint_path}")

    # Download deploy.yaml if it exists
    if deploy_yaml_file:
        params_dir = os.path.join(download_dir, "params")
        os.makedirs(params_dir, exist_ok=True)
        deploy_file = deploy_yaml_file.download(root=download_dir, replace=True)
        deploy_file.close()
        print(f"[INFO] Downloaded deploy.yaml to: {os.path.join(params_dir, 'deploy.yaml')}")
    else:
        print("[INFO] No deploy.yaml found in WandB run")

    return checkpoint_path


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Honor --run_name from cli_args for local loading
    if args_cli.run_name is not None:
        agent_cfg.load_run = args_cli.run_name

    # Initialize WandB if enabled
    wandb_run = None
    use_wandb = not args_cli.disable_wandb

    if use_wandb:
        wandb_project = getattr(agent_cfg, "wandb_project", "unitree_rl_lab")

        # Resume existing WandB run
        print(f"[INFO] Playing WandB run: {args_cli.run_path}")
        wandb_run = wandb.init(
            project=wandb_project,
            id=args_cli.run_path.split("/")[-1],
            resume="allow",
            mode="online",
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # Handle checkpoint loading
    if args_cli.run_path and use_wandb:
        print(f"[INFO] Downloading checkpoint from WandB run: {args_cli.run_path}")
        try:
            run_name = args_cli.run_path.split("/")[-1]
            wandb_ckpt_dir = os.path.join(log_root_path, f"wandb_{run_name}")
            resume_path = _download_latest_wandb_checkpoint(args_cli.run_path, wandb_ckpt_dir)

            if resume_path is not None:
                print(f"[INFO] Using checkpoint from WandB: {resume_path}")
            else:
                print("[WARNING] No model_*.pt checkpoint found in WandB run, falling back to local checkpoint")
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        except Exception as e:
            print(f"[WARNING] Failed to download from WandB: {e}")
            print("[INFO] Falling back to local checkpoint")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # params for deployment
    params_dir = os.path.join(os.path.dirname(resume_path), "params")

    # Upload exported policies to WandB
    if use_wandb and wandb_run is not None:
        print("[INFO] Uploading exported policies to WandB...")
        try:
            # Create artifact for exported models
            artifact = wandb.Artifact(
                name="exported_policy",
                type="model",
                description=f"Exported policy for {args_cli.task}",
            )
            # artifact.add_file(os.path.join(export_model_dir, "policy.pt"))
            artifact.add_file(os.path.join(export_model_dir, "policy.onnx"))
            artifact.add_file(os.path.join(params_dir, "deploy.yaml"))

            # Log artifact and wait for upload to complete
            logged_artifact = wandb_run.log_artifact(artifact)
            print("[INFO] Waiting for artifact upload to complete...")
            logged_artifact.wait()
            print("[INFO] Successfully uploaded exported policies to WandB")
        except Exception as e:
            print(f"[WARNING] Failed to upload to WandB: {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # Finish WandB run
    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        print("[INFO] WandB run finished")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
