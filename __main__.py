import Analyze
import score_funcs
import KMeans
import DBSCAN
import CompetitiveLearning
import PSO
import ACO
import load_data
from tkinter import *

# Parameters to use for each algorithm and dataset

# Number of centroids
kMeans_params = { "Iris" : [3],
                  "Glass" : [7],
                  "Banknote" : [2],
                  "Seeds" : [3],
                  "Customer" : [5]}

# num clusters, num ants, beta, prob cutoff, num elite, decay rate, q
aco_params = { "Iris" : [3],
               "Glass" : [7],
               "Banknote" : [2],
               "Seeds" : [3],
               "Customer" : [5]}

# num_particles, num centroids, inertia, accel 1, accel 2, max_iter
pso_params = { "Iris" : [5, 3, 0.8, 1.5, 1.3, 100],
               "Glass" : [7],
               "Banknote" : [2],
               "Seeds" : [3],
               "Customer" : [5]}

# eta, num_clusters, iterations
cl_params = { "Iris" : [0.1, 3, 100],
              "Glass" : [7],
              "Banknote" : [2],
              "Seeds" : [3],
              "Customer" : [5]}

# radius, minpts
dbscan_params = { "Iris" : [3],
                  "Glass" : [7],
                  "Banknote" : [2],
                  "Seeds" : [3],
                  "Customer" : [5]}


class build_GA_Menu(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.master.title("Clustering Analysis")
        self.pack(fill=BOTH, expand=1)

        # Drop down menu for dataset selection
        dataLabel = Label(self, text="Select Dataset")
        dataLabel.grid(row=0, column=0)
        options = ["Iris", "Seeds", "Glass", "Banknote", "Customers"]
        self.data_selection = StringVar(self.master)
        self.data_selection.set("            ")

        self.x = OptionMenu(self, self.data_selection, *options)
        self.x.grid(row=1, column=0)

        # Drop down menu for algorithm selection
        algLabel = Label(self, text="Select Algorithm")
        algLabel.grid(row=0, column=1)
        options = ["K-Means", "DBSCAN", "Competitive Learning", "PSO", "ACO"]
        self.alg_selection = StringVar(self.master)
        self.alg_selection.set("            ")

        self.x = OptionMenu(self, self.alg_selection, *options)
        self.x.grid(row=1, column=1)

        # Create button to run the algorithm
        classButton = Button(self, text="Run!", command=self.run)
        classButton.grid(row=1, column=2)

    def run(self):
        # First the selected dataset needs to be loaded
        dataset_name = self.data_selection.get()
        if dataset_name == "Iris":
            print ("Selecting Iris!")
            data = load_data.load_iris()
        elif dataset_name == "Seeds":
            data = load_data.load_seeds()
        elif dataset_name == "Glass":
            data = load_data.load_glass()
        elif dataset_name == "Banknote":
            data = load_data.load_banknote()
        elif dataset_name == "Customers":
            data = load_data.load_cust_data()

        # Now run the selected clustering algorithm
        score_list = [score_funcs.cluster_sse]
        if self.alg_selection.get() == "K-Means":
            Analyze.analyze(data, dataset_name, 10, self.build_kMeans_func(*kMeans_params[dataset_name]), score_list)
        elif self.alg_selection.get() == "DBSCAN":
            # Iris params: 0.5, 4
            # DBSCAN.parameter_selection(4, data)
            Analyze.analyze(data, dataset_name, 10, self.build_dbscan_func(*dbscan_params[dataset_name]), score_list)
        elif self.alg_selection.get() == "Competitive Learning":
            Analyze.analyze(data, dataset_name, 10, self.build_cl_func(*cl_params[dataset_name]), score_list)
        elif self.alg_selection.get() == "PSO":
            Analyze.analyze(data, dataset_name, 5, self.build_pso_function(*pso_params[dataset_name]), score_list)
        elif self.alg_selection.get() == "ACO":
            Analyze.analyze(data, dataset_name, 5, self.build_aco_func(iterations = 10, num_clusters = 3, num_ants = 8,
                            beta = 0.5, prob_cutoff = 0.75, num_elite_ants = 3, decay_rate = .75, q = 1), score_list)

    # pass the build function the arguments to the function
    def build_kMeans_func(self, k):
        # get the arguments:
        params = locals()

        # Create a run function that passes the dataset and scoring metrics that will be used to the given clustering
        # algorithm
        def run_function(dataset, score_funcs):
            return KMeans.kMeans(dataset, score_funcs, k)

        # The run_function is given parameters to simplify writing to file with detailed description of the algorithm
        # and parameters that were used in a given experiment
        run_function.params = params
        run_function.alg_name = "K-Means"
        return run_function

    def build_dbscan_func(self, radius, minpts):
        params = locals()

        # Create a run function that passes the dataset and scoring metrics that will be used to the given clustering
        # algorithm
        def run_function(dataset, score_funcs):
            return DBSCAN.dbscan(dataset, radius, minpts, score_funcs=score_funcs)

        # The run_function is given parameters to simplify writing to file with detailed description of the algorithm
        # and parameters that were used in a given experiment
        run_function.params = params
        run_function.alg_name = "DBSCAN"
        return run_function

    def build_cl_func(self, eta, num_clusters, iterations):
        params = locals()

        # Create a run function that passes the dataset and scoring metrics that will be used to the given clustering
        # algorithm
        def run_function(dataset, score_funcs):
            return CompetitiveLearning.competitive_learning(dataset, eta, num_clusters, iterations, score_funcs)

        # The run_function is given parameters to simplify writing to file with detailed description of the algorithm
        # and parameters that were used in a given experiment
        run_function.params = params
        run_function.alg_name = "Competitive Learning"
        return run_function

    def build_pso_function(self, num_particles, num_centroids, inertia, accel_1, accel_2, max_iter):
        params = locals()

        # Create a run function that passes the dataset and scoring metrics that will be used to the given clustering
        # algorithm
        def run_function(dataset, score_funcs):
            return PSO.pso(num_particles, num_centroids, inertia, accel_1, accel_2, max_iter, dataset, score_funcs=score_funcs)

        # The run_function is given parameters to simplify writing to file with detailed description of the algorithm
        # and parameters that were used in a given experiment
        run_function.params = params
        run_function.alg_name = "PSO"
        return run_function

    def build_aco_func(self, iterations, num_clusters, num_ants, beta, prob_cutoff, num_elite_ants, decay_rate, q):
        params = locals()

        # Create a run function that passes the dataset and scoring metrics that will be used to the given clustering
        # algorithm
        def run_function(dataset, score_funcs):
            return ACO.ACO(dataset, iterations, num_clusters, num_ants, beta, prob_cutoff, num_elite_ants, decay_rate, q, score_funcs)

        # The run_function is given parameters to simplify writing to file with detailed description of the algorithm
        # and parameters that were used in a given experiment
        run_function.params = params
        run_function.alg_name = "ACO"
        return run_function

if __name__ == '__main__':
    root = Tk()
    app = build_GA_Menu(root)
    root.mainloop()
