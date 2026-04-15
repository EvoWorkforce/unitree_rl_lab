#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to download policy.onnx from WandB run."""

import argparse
import os
import re
import shutil
import wandb
import yaml


def get_used_keys(config_data):
    """Extract all used key bindings from the config."""
    used_keys = set()
    if 'all_transitions' in config_data:
        for state_name, binding in config_data['all_transitions'].items():
            if isinstance(binding, str) and 'key_' in binding:
                # Extract key from "key_x.on_pressed"
                match = re.search(r'key_(\w+)\.', binding)
                if match:
                    used_keys.add(match.group(1))
    return used_keys


def get_next_fsm_id(config_data, id_range_start):
    """Find the next available FSM ID in the given range."""
    used_ids = set()
    if 'FSM' in config_data and '_' in config_data['FSM']:
        for state_name, state_config in config_data['FSM']['_'].items():
            if isinstance(state_config, dict) and 'id' in state_config:
                used_ids.add(state_config['id'])
    
    # Find next available ID starting from id_range_start
    next_id = id_range_start
    while next_id in used_ids:
        next_id += 1
    return next_id


def update_config_yaml(config_path, run_name, policy_type):
    """Update config.yaml to add new FSM state for the deployed policy."""
    print(f"\n[INFO] Updating config.yaml to add FSM state for '{run_name}'...")
    
    # Read the YAML file preserving structure
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Parse YAML
    config_data = yaml.safe_load(config_content)
    
    # Check if state already exists
    if 'FSM' in config_data and '_' in config_data['FSM']:
        if run_name in config_data['FSM']['_']:
            print(f"[WARNING] FSM state '{run_name}' already exists in config.yaml")
            overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("[INFO] Skipping config.yaml update")
                return
    
    # Get used keys
    used_keys = get_used_keys(config_data)
    print(f"[INFO] Currently used keys: {sorted(used_keys)}")
    
    # Ask user for key binding
    while True:
        key = input(f"Enter an alphanumeric key for '{run_name}' transition: ").strip().lower()
        if not key or not re.match(r'^[a-z0-9]$', key):
            print("[ERROR] Please enter a single alphanumeric character (a-z, 0-9)")
            continue
        if key in used_keys:
            print(f"[ERROR] Key '{key}' is already assigned. Please choose another key.")
            continue
        break
    
    # Get next available ID (for pose_tracking, use 2XX range)
    if policy_type == "velocity":
        next_id = get_next_fsm_id(config_data, 101)
    elif policy_type == "pose_tracking":
        next_id = get_next_fsm_id(config_data, 201)
    elif policy_type == "mimic":
        next_id = get_next_fsm_id(config_data, 301)
    else:
        next_id = get_next_fsm_id(config_data, 401)
    
    print(f"[INFO] Assigning ID: {next_id}, Key: {key}")
    
    # Update config content by inserting the new entries
    # 1. Add to all_transitions at the end of the list
    # Find the last line of all_transitions (before the next YAML section)
    transitions_pattern = r'(all_transitions: &all_transitions\n(?:  \w+:.*\n)+)'
    
    def add_transition(match):
        existing = match.group(1)
        new_transition = f'  {run_name}: key_{key}.on_pressed\n'
        return existing + new_transition
    
    config_content = re.sub(transitions_pattern, add_transition, config_content)
    
    # 2. Add to FSM._ definitions at the end
    # Find the FSM._ section and add at the end before the first state definition (before next line that doesn't have indent)
    type_mapping = {
        "pose_tracking": "PoseTracking",
        "velocity": "RLBase",
        "mimic": "Mimic"
    }
    state_type = type_mapping.get(policy_type, "Unknown")
    
    # Pattern to find end of FSM._ section (before the next top-level key that starts without 4 spaces)
    fsm_defs_pattern = r'(  _: # enabled fsms\n(?:    \w+:\n(?:      \w+:.*\n)+)+)'
    
    def add_fsm_def(match):
        existing = match.group(1)
        new_fsm_def = f'''    {run_name}:
      id: {next_id}
      type: {state_type}
'''
        return existing + new_fsm_def
    
    config_content = re.sub(fsm_defs_pattern, add_fsm_def, config_content)
    
    # 3. Add state configuration at the end of the file
    extra_fields = ""
    if policy_type == "pose_tracking":
        extra_fields = "\n    goal_pose_topic: rt/goal_pose\n    # debug_print: true"
    new_state_config = f'''\n
  {run_name}:
    transitions: *all_transitions
    policy_dir: config/policy/{policy_type}/{run_name}{extra_fields}
'''
    config_content = config_content.rstrip() + new_state_config
    
    # Write back to file
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"[SUCCESS] Updated config.yaml with new FSM state '{run_name}'")
    print(f"[SUCCESS] Key binding: {key}, FSM ID: {next_id}")


def main():
    """Download policy.onnx from WandB run to local policies directory."""
    parser = argparse.ArgumentParser(description="Download policy.onnx from WandB run.")
    parser.add_argument(
        "--run_path",
        type=str,
        required=True,
        help="WandB run path (format: entity/project/run_id) to download policy from.",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        choices=["velocity", "mimic", "pose_tracking"],
        help="Type of policy to download (velocity, mimic, or pose_tracking).",
    )
    args = parser.parse_args()

    print(f"[INFO] Downloading policy from WandB run: {args.run_path}")
    
    try:
        # Initialize WandB API
        api = wandb.Api()
        
        # Get the run
        run = api.run(args.run_path)
        print(f"[INFO] Connected to run: {run.name}")
    
        # Create output directory
        policy_dir = 'deploy/robots/g1_23dof/config/policy'
        output_dir = os.path.join(policy_dir, args.policy_type, run.name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory: {output_dir}")
        
        # List all artifacts for this run
        artifacts = list(run.logged_artifacts())
        
        # Find all exported_policy artifacts
        exported_policy_artifacts = []
        for artifact in artifacts:
            if artifact.type == "model" and "exported_policy" in artifact.name:
                exported_policy_artifacts.append(artifact)
        
        if not exported_policy_artifacts:
            print("[ERROR] No exported_policy artifact found in this run.")
            print("[INFO] Available artifacts:")
            for artifact in artifacts:
                print(f"  - {artifact.name} (type: {artifact.type})")
            return 1
        
        # Sort by creation time and get the latest
        exported_policy_artifacts.sort(key=lambda x: x.created_at, reverse=True)
        exported_policy_artifact = exported_policy_artifacts[0]
        
        print(f"[INFO] Found {len(exported_policy_artifacts)} exported_policy artifact(s)")
        print(f"[INFO] Using latest artifact: {exported_policy_artifact.name} (created: {exported_policy_artifact.created_at})")
        
        # Download the artifact
        artifact_dir = exported_policy_artifact.download()
        print(f"[INFO] Downloaded to: {artifact_dir}")
        
        # Copy policy.onnx to output directory
        exported_dir = os.path.join(output_dir, "exported")
        os.makedirs(exported_dir, exist_ok=True)
        onnx_source = os.path.join(artifact_dir, "policy.onnx")
        onnx_dest = os.path.join(exported_dir, "policy.onnx")
        
        if os.path.exists(onnx_source):
            shutil.copy2(onnx_source, onnx_dest)
            print(f"[SUCCESS] Policy saved to: {onnx_dest}")
        else:
            print(f"[ERROR] policy.onnx not found in artifact directory: {artifact_dir}")
            print("[INFO] Files in artifact:")
            for file in os.listdir(artifact_dir):
                print(f"  - {file}")
            return 1
        
        # Copy deploy.yaml to output directory
        params_dir = os.path.join(output_dir, "params")
        os.makedirs(params_dir, exist_ok=True)
        yaml_source = os.path.join(artifact_dir, "deploy.yaml")
        yaml_dest = os.path.join(params_dir, "deploy.yaml")
        
        if os.path.exists(yaml_source):
            shutil.copy2(yaml_source, yaml_dest)
            print(f"[SUCCESS] Deploy config saved to: {yaml_dest}")
        else:
            print(f"[ERROR] deploy.yaml not found in artifact directory: {artifact_dir}")
            print("[INFO] Files in artifact:")
            for file in os.listdir(artifact_dir):
                print(f"  - {file}")
            return 1
        
        # Modify config.yaml to add new FSM state for the deployed policy
        config_path = "deploy/robots/g1_23dof/config/config.yaml"
        if not os.path.exists(config_path):
            print(f"[WARNING] Config file not found: {config_path}")
            print("[WARNING] Skipping config.yaml update")
        else:
            update_config_yaml(config_path, run.name, args.policy_type)

        return 0
        
    except Exception as e:
        print(f"[ERROR] Failed to download from WandB: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
