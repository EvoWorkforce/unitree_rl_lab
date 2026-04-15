import os
import sys
import shutil
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Export policy script.")
    parser.add_argument('--log_dir', required=True, help='Relative path to the log directory')
    parser.add_argument('--policy_name', required=True, help='Name for the exported policy directory')
    args = parser.parse_args()

    dest_dir_name = re.sub(r'(?<!^)([A-Z])', r'_\1', args.policy_name).lower()

    log_dir = os.path.abspath(args.log_dir)
    exported_dir = os.path.join(log_dir, 'exported')
    params_dir = os.path.join(log_dir, 'params')

    if not os.path.isdir(exported_dir):
        print(f"Error: '{exported_dir}' directory does not exist.")
        sys.exit(1)
    if not os.path.isdir(params_dir):
        print(f"Error: '{params_dir}' directory does not exist.")
        sys.exit(1)

    dest_base = os.path.join(os.path.dirname(__file__), '../../deploy/robots/g1_23dof/config/policy/mimic')
    dest_base = os.path.abspath(dest_base)
    dest_dir = os.path.join(dest_base, dest_dir_name)

    if os.path.exists(dest_dir):
        print(f"Error: Destination directory '{dest_dir}' already exists.")
        sys.exit(1)

    os.makedirs(dest_dir)
    shutil.copytree(exported_dir, os.path.join(dest_dir, 'exported'), dirs_exist_ok=True)
    shutil.copytree(params_dir, os.path.join(dest_dir, 'params'), dirs_exist_ok=True)
    print(f"Policy exported to {dest_dir}")

if __name__ == '__main__':
    main()
