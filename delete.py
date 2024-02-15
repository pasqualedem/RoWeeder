import concurrent.futures
import wandb

api = wandb.Api()
runs = api.runs(path="pasqualedem/weedmapping-rededge", per_page=300)

def delete_run_files(run):
    for file in run.files():
        if "media" in file.name:
            file.delete()
    return run.id

def delete_run_artifacts(run):
    for artifact in run.logged_artifacts():
        if "history" in artifact.name:
            continue
        artifact.delete(delete_aliases=True)
    return run.id

def main():
    # Create a ThreadPoolExecutor with, for example, 4 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Use submit() to submit the function for each iteration
        # The results are stored in a list of Future objects
        futures = [executor.submit(delete_run_artifacts, run) for run in runs]

        # Use as_completed() to iterate over the completed futures
        for future in concurrent.futures.as_completed(futures):
            try:
                # Retrieve the result of each future
                result = future.result()
                print(f"Result: {result}")
            except Exception as e:
                print(f"An exception occurred: {e}")

if __name__ == "__main__":
    main()
