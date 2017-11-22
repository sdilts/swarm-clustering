# This file defines two functions for analyzing the output of the clustering functions.
import os, getpass, errno
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

def _save_run_info(save_loc, alg_func, dataset):
    with open(os.path.join(save_loc, "func_info.txt"), 'w') as file:
        file.write("Alg Name: " + alg_func.alg_name + "\n")
        file.write("Dataset: " + dataset + "\n")
        file.write("Params:\n")
        for key in alg_func.params:
            file.write("\t" + key + ": " + str(alg_func.params[key]))
        file.write("\n")



def analyze(dataset, dataset_name, repeat,alg_func, score_funcs):
    save_loc = _create_output_folder(dataset_name, alg_func.alg_name)
    _save_run_info(save_loc, alg_func, dataset_name)
    iteration_results = []
    final_states = []

    for i in range(repeat):
        results = alg_func(dataset, score_funcs)
        # assumes that the last item in the list is the final result:
        final_states.append(results[-1])
        iteration_results.append(results)


    # do some anaylisis on the data:


    # save the stuff into a file...

def analyze_clusters(clusters, score_funcs):
    """Analyzes the cluster based on the score functions given.
    """
    results = dict()
    for func in score_funcs:
        results[func.name] = func(clusters)
    return results
