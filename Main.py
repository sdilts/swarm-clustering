# import Analyze
# import score_funcs
import KMeans
import DBSCAN
import CompetitiveLearning
import os
import random


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


def build_dbscan_func(radius, minpts):
    params = locals()

    def run_function(dataset, score_funcs):
        return DBSCAN.dbscan(dataset, radius, minpts, score_funcs=score_funcs)

    run_function.params = params
    run_function.alg_name = "DBSCAN"
    return run_function

def build_cl_function(eta, num_clusters, iterations):
    params = locals()

    def run_function(dataset, score_funcs):
        return CompetitiveLearning.competitive_learning(dataset, eta, num_clusters, iterations, score_funcs)

    run_function.params = params
    run_function.alg_name = "Competitive Learning"
    return run_function

if __name__ == '__main__':
    # score_list = [score_funcs.score_1, score_funcs.score_2]
    # dataset = [(1,1), (2,2), (10,10), (11,11)]
    # Analyze.analyze(dataset, "test", 10, build_kMeans_func(2), score_list)

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Ex_Clusters', 'globular.txt')
    print(dir_path)

    # data_pts = DBSCAN.load_data(dir_path)
    pts1 = [(random.uniform(0, 1), random.uniform(0, 1)) for x in range(0, 5)]
    pts2 = [(random.uniform(5, 6), random.uniform(5, 6)) for x in range(0, 5)]
    pts3 = [(random.uniform(1, 2), random.uniform(4, 5)) for x in range(0, 5)]
    data_pts = pts1 + pts2 + pts3

    # DBSCAN.parameter_selection(2, data_pts)

    # data, radius, minpts
    ans = DBSCAN.dbscan(data_pts, 0.5, 2)
    print(ans)
