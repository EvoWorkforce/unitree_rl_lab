import csv

# Indices of joints in the original 29dof joint_ids_map
original_joint_ids_map = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
# Indices to keep (those present in the new 23dof joint_ids_map)
new_joint_ids_map = [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26]

# Find indices to remove (in the order of the original_joint_ids_map)
remove_joint_ids = [jid for jid in original_joint_ids_map if jid not in new_joint_ids_map]
remove_indices = [original_joint_ids_map.index(jid) for jid in remove_joint_ids]

input_csv = 'G1_Take_102.bvh_60hz.csv'
output_csv = 'G1_Take_102.bvh_60hz_23dof.csv'

with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        # Remove columns in reverse order to avoid index shift
        for idx in sorted(remove_indices, reverse=True):
            if idx < len(row):
                del row[idx]
        writer.writerow(row)

print(f"Saved reduced CSV to {output_csv}")
