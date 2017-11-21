import Analyze
import score_funcs
import kNN


def main():

    # example for running an anaylisis of the k-NN algorithm
    # Might add another class to make this less messy, but the
    # ICluster interface must also define a dictionary called params, which are
    # the parameters that are passed into the algorithm function:
    def run_function(dataset, score_funcs):
        run_function.params = {"k" : 1}
        run_function.alg_name = "k-NN"
        return kNN.cluster(dataset, score_funcs, params["k"])

    score_list = [score_funcs.score_1, score_funcs.score_2]

    Analyze.analyze(dataset, 10, run_function, score_list)
