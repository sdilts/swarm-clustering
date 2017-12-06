# This file defines two functions for analyzing the output of the clustering functions.
import os, getpass, errno
import pandas as pd
import time
import DelayedKeyboardInterrupt as dk

'''This module contains the functionality to run experiments from beginning to end including performing the desired
   number of tests, as well as tracking the cluster scores at each iteration, and finally saving all results and
   statistics to file'''


# Method to make a new directory to hold the results of the experiment
def _create_output_folder(dataset_name, alg_name):
    # Returns the folder where the output files should be saved.
    user = getpass.getuser()
    time_start = time.strftime("%m_%d_%H_%M_%S")

    folder_dir = os.path.abspath("./outputs")

    # Make output directory
    try:
        os.makedirs(folder_dir)
        print("Output directory created at",
              os.path.relpath(folder_dir, os.path.dirname(os.path.abspath(__file__))))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # saving stuff needed throughout the whole function:
    output_dir = os.path.join(folder_dir, dataset_name + "_" + alg_name + "_" + user + time_start)

    if os.path.exists(output_dir):
        print("WARNING: Output folder already exists. Saving in backup location. The next time this error occurs, this location will be overwritten")
        output_file = os.path.join(folder_dir, "backup_dir")
    else:
        # Make output for this run:
        try:
            os.makedirs(output_dir)
            print("Data directory created at",
                  os.path.relpath(output_dir,os.path.dirname(os.path.abspath(__file__))))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return output_dir


# Method to save the algorithm parameters used in an experiment
def _save_run_info(save_loc, alg_func, score_funcs, dataset):
    # Extracts the metadata from the given parameters and saves them to a file
    # called "run_info.txt"

    with open(os.path.join(save_loc, "run_info.txt"), 'w') as file:
        file.write("Alg Name: " + alg_func.alg_name + "\n")
        file.write("Dataset: " + dataset + "\n")
        file.write("Scoring functions:")

        for func in score_funcs:
            file.write(" " + func.name)
        file.write("\nParams:\n")

        for key in alg_func.params:
            file.write("\t" + key + ": " + str(alg_func.params[key]))
        file.write("\n")


# Method to calculate statistics from the experiments
def _compute_statistics(data_frame):
    stats = dict()
    stats["stdev"] = data_frame.std()
    stats["mean"] = data_frame.mean()
    stats["max"] = data_frame.max()
    stats["min"] = data_frame.min()
    return pd.DataFrame(stats)


# Method to save all cluster scores at each iteration as well as convergence data
def _save_run_data(save_loc,final_states, iteration_results):
    print("\nSaving Data\n-----------------------")

    # Convert all of our final vaules into pandas dataframes:
    final_table = pd.DataFrame(final_states)

    # Perform analysis on the data:
    stats = _compute_statistics(final_table)

    # save the file as the summary:
    print("Saving final data... ", end='')
    stats.to_csv(os.path.join(save_loc, "final_stats.csv"), index=True, header=True)
    final_table.to_csv(os.path.join(save_loc, "final_data.csv"), header=True)
    print("Done.")

    # Only save iteration data if more than one iteration was required:
    if any(len(result) > 1 for result in iteration_results):
        print("Saving iteration data..",end='')

        # convert all iteration data into pandas dataframes:
        iteration_tables = [pd.DataFrame(results) for results in iteration_results]
        for i, tbl in enumerate(iteration_tables):
            tbl.to_csv(os.path.join(save_loc, "iteration" + str(i) + "_data.csv"), header=True)

        print(" Done")
        print("Saving convergence data..", end='')

        # the number of rows should be the number of iterations it took to converge:
        convg_data = pd.DataFrame([tbl.shape[0] for tbl in iteration_tables],columns=["iterations to convergence"])
        convg_data.to_csv(os.path.join(save_loc, "convg_data.csv"), index=True, header=True)
        convg_stats = _compute_statistics(convg_data)
        convg_stats.to_csv(os.path.join(save_loc, "convg_stats.csv"), index=True, header=True)
        print(" Done")

    else:
        print("Iteration data same as final data. Not saving.")


# Method to actually run the tests and call the save functions
def analyze(dataset, dataset_name, repeat, alg_func, score_funcs):
    # Create save directories
    save_loc = _create_output_folder(dataset_name, alg_func.alg_name)
    _save_run_info(save_loc, alg_func, score_funcs, dataset_name)
    iteration_results = []
    final_states = []

    # Try to run the tests
    try:
        print("\nRunning Tests\n-----------------------")
        for i in range(repeat):
            print("Running test",i+1,"out of", repeat,"...")
            results = alg_func(dataset, score_funcs)
            with dk.DelayedKeyboardInterrupt():
                # assumes that the last item in the list is the final result:
                final_states.append(results[-1])
                iteration_results.append(results)
    except KeyboardInterrupt:
        print("\033[1;31m\nTesting terminated prematurely\n\033[0m\n")

    with dk.DelayedKeyboardInterrupt():
        _save_run_data(save_loc, final_states,iteration_results)

    print("\nAll finished.\nHave a nice day!\n")


# Return the sse score of the provided cluster
def analyze_clusters(clusters, score_fns):
    """Analyzes the cluster based on the score functions given.
    """
    results = dict()
    for func in score_fns:
        results[func.name] = func(clusters)
    return results
