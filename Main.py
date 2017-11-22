import Analyze
import score_funcs
import KMeans



# pass the build function the arguments to the function
def build_kMeans_func(k):
    # get the arguments:
    params = locals()

    # example for running an anaylisis of the k-NN algorithm
    # Might add another class to make this less messy, but the
    # ICluster interface must also define a dictionary called params, which are
    # the parameters that are passed into the algorithm function:
    def run_function(dataset, score_funcs):
        return KMeans.kMeans(dataset, score_funcs, k)

    run_function.params = params
    run_function.alg_name = "K-Means"
    return run_function

if __name__ == '__main__':
    score_list = [score_funcs.score_1, score_funcs.score_2]
    dataset = [(1,1), (2,2), (10,10), (11,11)]
    Analyze.analyze(dataset, "test", 10, build_kMeans_func(2), score_list)
