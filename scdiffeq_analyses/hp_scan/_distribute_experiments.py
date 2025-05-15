# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, List


# -- function: ----------------------------------------------------------------
def distribute_experiments(
    experiments: List[Dict[str, Any]], num_vms: int
) -> Dict[str, List[int]]:
    """
    Distribute experiments across VMs.

    Args:
        experiments: List of experiment configurations
        num_vms: Number of VMs to distribute experiments across

    Returns:
        Dictionary mapping VM names to lists of experiment IDs
    """
    vm_to_experiments = {}
    num_experiments = len(experiments)

    # Calculate number of experiments per VM
    experiments_per_vm = (num_experiments + num_vms - 1) // num_vms  # Ceiling division

    for i in range(num_vms):
        vm_name = f"scdiffeq-vm-{i+1}"
        start_idx = i * experiments_per_vm
        end_idx = min((i + 1) * experiments_per_vm, num_experiments)

        vm_experiments = [experiments[j]["id"] for j in range(start_idx, end_idx)]
        if vm_experiments:  # Only add VMs that have experiments to run
            vm_to_experiments[vm_name] = vm_experiments

    return vm_to_experiments
