# This file defines two functions for analyzing the output of the clustering functions.
import os

def _create_ouptut_folder(alg_name):
    """Returns the folder where the output files should be saved.
    """
    user = getpass.getuser()
    time_start = time.strftime("%m_%d_%H_%M_%S")

    folder_dir = os.path.abspath("./outputs")

    # Make output directory
    try:
        os.makedirs(folder_dir)
        print("Output directory created at " + folder_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
        raise

    # saving stuff needed throughout the whole function:
    output_folder = os.path.join(folder_dir, alg_name + user + time_start)

    if os.path.exists(output_dir):
        print("WARNING: Output folder already exists. Saving in backup location. The next time this error occurs, this location will be overwritten")
        output_folder = os.path.join(folder_dir, "backput_dir")
    else:
        # Make output for this run:
        try:
            os.makedirs(output_dir)
            print("Data directory created at " + folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
            raise
    return output_folder

def _save_func_info(save_loc, alg_func):
    with open(os.path.join(save_loc, "func_info.txt"), 'w') as file:
        file.write("Alg Name " + alg_func.alg_name)
        file.write("Params:")
        for key in alg_func.params:
            file.write("\t" + key + alg_func.params[key])



def analyze(dataset, repeat,alg_func, score_funcs):

    save_loc = _create_output_folder(alg_func.alg_name)
    _save_func_info(save_loc, alg_func)

    # panda stuff goes here:
    for i in range(repeat):
    #     results = alg_func(dataset, score_funcs)
        _save_table(i, results)


    # do some anaylisis on the data:


    # save the stuff into a file...

def analyze_cluster(cluster, score_funcs):
    """Analyzes the cluster based on the score functions given
    pass
