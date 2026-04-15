import os
import sys
import shutil
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Export policy script.")
    parser.add_argument('--locomotion', action='store_true', default=False, help='Export locomotion policy')
    parser.add_argument('--mimic', action='store_true', default=False, help='Export mimic policy')
    parser.add_argument('--task', required=True, help='Task name')
    parser.add_argument('--run_name', type=str, default=None, help='Run name, <DATE>_<TIME> format, otherwise latest is used')
    parser.add_argument('--output_dir', type=str, default=None, help='Name for the exported policy directory, otherwise derived from task name')
    args = parser.parse_args()

    if not args.locomotion and not args.mimic:
        print("Error: Please specify either --locomotion or --mimic to export the respective policy.")
        sys.exit(1)

    task_dir = args.task.replace('-', '_').lower()
    log_dir = os.path.join('logs', 'rsl_rl', task_dir)

    # Use task name to derive output directory name if not provided
    if args.output_dir:
        dest_dir_name = args.output_dir
    else:
        prefix = "unitree_g1_23dof_"
        dest_dir_name = task_dir[len(prefix):] if task_dir.startswith(prefix) else task_dir

    # Find the latest run directory if not specified
    run_name = args.run_name
    if not run_name:
        all_runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        date_time_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
        valid_runs = [d for d in all_runs if date_time_pattern.match(d)]
        if not valid_runs:
            print(f"No valid runs found in {log_dir}")
            sys.exit(1)
        run_name = max(valid_runs)

    print(f"Exporting policy from run: {run_name}")
    log_dir = os.path.join(log_dir, run_name)

    exported_src_dir = os.path.join(log_dir, 'exported')
    params_src_dir = os.path.join(log_dir, 'params')

    if not os.path.isdir(exported_src_dir):
        print(f"Error: '{exported_src_dir}' directory does not exist.")
        sys.exit(1)
    if not os.path.isdir(params_src_dir):
        print(f"Error: '{params_src_dir}' directory does not exist.")
        sys.exit(1)

    dest_base = os.path.join('deploy/robots/g1_23dof/config/policy')

    if args.locomotion:
        dest_base = os.path.join(dest_base, 'velocity')
    elif args.mimic:
        dest_base = os.path.join(dest_base, 'mimic')

    dest_base = os.path.abspath(dest_base)
    dest_dir = os.path.join(dest_base, dest_dir_name)

    if os.path.exists(dest_dir):
        print(f"Destination directory already exists: {dest_dir}\nAre you sure you want to overwrite it? (y/n): ", end='')
        choice = input().lower()
        if choice != 'y':
            print("Aborting export")
            sys.exit(0)

    exported_dest_dir = os.path.join(dest_dir, 'exported')
    params_dest_dir = os.path.join(dest_dir, 'params')

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(exported_dest_dir, exist_ok=True)
    os.makedirs(params_dest_dir, exist_ok=True)
    shutil.copy(os.path.join(exported_src_dir, 'policy.onnx'), exported_dest_dir)
    shutil.copy(os.path.join(params_src_dir, 'deploy.yaml'), params_dest_dir)

    deploy_cfg_file = os.path.join(params_dest_dir, 'deploy.yaml')

    # Add to the top of deploy.yaml for mimic policies
    # joint_policy_map: [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
    # joint_ids_map: [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26]

    if args.mimic:
        lines = []
        with open(deploy_cfg_file, 'r') as f:
            lines = f.readlines()
        with open(deploy_cfg_file, 'w') as f:
            f.write('joint_policy_map: [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]\n')
            f.write('joint_ids_map: [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26]\n')
            f.writelines(lines[2:])

    print(f"Policy exported to: {dest_dir}")

    if args.mimic:
        src_dir_base = os.path.join('source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof')
        task_name = args.task
        found_csv = False
        for subdir in os.listdir(src_dir_base):
            subdir_path = os.path.join(src_dir_base, subdir)
            init_path = os.path.join(subdir_path, '__init__.py')
            if os.path.isdir(subdir_path) and os.path.isfile(init_path):
                with open(init_path, 'r') as f:
                    if task_name in f.read():
                        # Find .csv file in this directory
                        for file in os.listdir(subdir_path):
                            if file.endswith('.csv'):
                                shutil.copy(os.path.join(subdir_path, file), params_dest_dir)
                                print(f"Copied motion file: {file}")
                                found_csv = True
                                break
                if found_csv:
                    break
        if not found_csv:
            print(f"No .csv file found for task '{task_name}' in {src_dir_base}")

if __name__ == '__main__':
    main()
