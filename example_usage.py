from scdiffeq_analyses.wandb import WandbClient
import logging

# Configure basic logging to see output from the WandbClient
logging.basicConfig(level=logging.INFO)

def main():
    # Replace with your actual project and entity if different
    project_name = "your_project_name"  # FIXME: Replace with your project name
    entity_name = "your_entity_name"    # FIXME: Replace with your entity name (or remove if default)

    print(f"Initializing WandbClient for project: {entity_name}/{project_name}")
    
    try:
        # 1. Initialize WandbClient
        # If your entity is the default "scDiffEq" you can omit it:
        # client = WandbClient(project=project_name)
        client = WandbClient(project=project_name, entity=entity_name)

        # 2. Fetch Runs
        print("\nFetching runs from W&B...")
        client.get_runs() # This will log the number of runs found

        # 3. Format Run Summary
        # You can use the default minimum of 5 benchmarked checkpoints
        print("\nFormatting run summary (default min_benchmarked_ckpts=5)...")
        summary_lines_default = client.format_run_summary()
        
        if summary_lines_default:
            print("\n--- Run Summary (Default) ---")
            for line in summary_lines_default:
                print(line)
        else:
            print("No runs met the default criteria for the summary.")

        # Or specify a different minimum, e.g., 2
        custom_min_ckpts = 2
        print(f"\nFormatting run summary (min_benchmarked_ckpts={custom_min_ckpts})...")
        summary_lines_custom = client.format_run_summary(min_benchmarked_ckpts=custom_min_ckpts)

        if summary_lines_custom:
            print(f"\n--- Run Summary (Min {custom_min_ckpts} Checkpoints) ---")
            for line in summary_lines_custom:
                print(line)
        else:
            print(f"No runs met the custom criteria (min {custom_min_ckpts} checkpoints) for the summary.")

        # You can also access the lists of runs directly if needed
        print(f"\nTotal runs processed: {len(client._all_runs)}")
        print(f"Complete runs: {len(client._complete_runs)}")
        print(f"Failed runs: {len(client._failed_runs)}")
        if client._failed_runs:
            print("Details of failed runs:")
            for failed_run in client._failed_runs:
                print(f"  ID: {failed_run['id']}, Name: {failed_run['name']}, Error: {failed_run['error']}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print(f"An error occurred. Please check the logs. You might need to log in to W&B (wandb login) or check your project/entity names.")

if __name__ == "__main__":
    main() 