# This file defines two functions for analyzing the output of the clustering functions.
import os, getpass, errno
import pandas as pd
import time

def _create_output_folder(dataset_name, alg_name):
    """Returns the folder where the output files should be saved.
    """
    user = getpass.getuser()
    time_start = time.strftime("%m_%d_%H:%M.%S")

    folder_dir = os.path.abspath("./outputs")

    # Make output directory
    try:
        os.makedirs(folder_dir)
        print("Output directory created at " + folder_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # saving stuff needed throughout the whole function:
    output_dir = os.path.join(folder_dir, dataset_name + "_" + alg_name + "_" + user + time_start)

    if os.path.exists(output_dir):
        print("WARNING: Output folder already exists. Saving in backup location. The next time this error occurs, this location will be overwritten")
        output_folder = os.path.join(folder_dir, "backup_dir")
    else:
        # Make output for this run:
        try:
            os.makedirs(output_dir)
            print("Data directory created at " + output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return output_dir

def _save_run_info(save_loc, alg_func, score_funcs, dataset):
    """Extracts the metadata from the given parameters and saves them to a file
    called "run_info.txt"
    """
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

def _compute_statistics(data_frame):
    stats = dict()
    stats["stdev"] = data_frame.std()
    stats["mean"] = data_frame.mean()
    stats["max"] = data_frame.max()
    stats["min"] = data_frame.min()
    return pd.DataFrame(stats)

def analyze(dataset, dataset_name, repeat,alg_func, score_funcs):
    save_loc = _create_output_folder(dataset_name, alg_func.alg_name)
    _save_run_info(save_loc, alg_func, score_funcs, dataset_name)
    iteration_results = []
    final_states = []

    for i in range(repeat):
        results = alg_func(dataset, score_funcs)
        # assumes that the last item in the list is the final result:
        final_states.append(results[-1])
        iteration_results.append(results)

    #convert all of our final vaules into pandas dataframes:
    final_table = pd.DataFrame(final_states)
    # do some anaylisis on the data:
    stats = _compute_statistics(final_table)
    # save the file as the summary:
    print("Saving final data...")
    stats.to_csv(os.path.join(save_loc, "final_stats.csv"), index=True, header=True)
    final_table.to_csv(os.path.join(save_loc, "final_data.csv"), header=True)
    print("Done.")
    # only do the iteration data if it is different than that final data:
    if len(iteration_results[0]) > 0:
        print("Saving iteration data..")
        #convert all iteration data into pandas dataframes:
        iteration_tables = [pd.DataFrame(results) for results in iteration_results]
        for i, tbl in enumerate(iteration_tables):
            tbl.to_csv(os.path.join(save_loc, "iteration" + str(i) + "_data.csv"), header=True)
        print("Done")
    else:
        print("Iteration data same as final data. Not saving.")


def analyze_clusters(clusters, score_funcs):
    """Analyzes the cluster based on the score functions given.
    """
    results = dict()
    for func in score_funcs:
        results[func.name] = func(clusters)
    return results