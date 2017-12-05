import Analyze
import score_funcs
import KMeans
import DBSCAN
import CompetitiveLearning
import PSO
import load_data
import os
import random
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt


class build_GA_Menu(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        #self.init_gui()

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
        if self.data_selection.get() == "Iris":
            print ("Selecting Iris!")
            data = load_data.load_iris()
        elif self.data_selection.get() == "Seeds":
            data = load_data.load_seeds()
        elif self.data_selection.get() == "Glass":
            data = load_data.load_glass()
        elif self.data_selection.get() == "Banknote":
            data = load_data.load_banknote()
        elif self.data_selection.get() == "Customers":
            data = load_data.load_cust_data()

        # Now run the selected clustering algorithm
        score_list = [score_funcs.cluster_sse]
        if self.alg_selection.get() == "K-Means":
            Analyze.analyze(data, self.data_selection.get(), 10, self.build_kMeans_func(2), score_list)
        elif self.alg_selection.get() == "DBSCAN":
            # Iris params: 0.5, 4
            # DBSCAN.parameter_selection(4, data)
            Analyze.analyze(data, self.data_selection.get(), 10, self.build_dbscan_func(0.5, 4), score_list)
        elif self.alg_selection.get() == "Competitive Learning":
            Analyze.analyze(data, self.data_selection.get(), 10, self.build_cl_func(0.1, 3, 100), score_list)
        elif self.alg_selection.get() == "PSO":
            Analyze.analyze(data, self.data_selection.get(), 5, self.build_pso_function(5, 3, 0.8, 1.5, 1.3, 100), score_list)

    # pass the build function the arguments to the function
    def build_kMeans_func(self, k):
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

    def build_dbscan_func(self, radius, minpts):
        params = locals()

        def run_function(dataset, score_funcs):
            return DBSCAN.dbscan(dataset, radius, minpts, score_funcs=score_funcs)

        run_function.params = params
        run_function.alg_name = "DBSCAN"
        return run_function

    def build_cl_func(self, eta, num_clusters, iterations):
        params = locals()

        def run_function(dataset, score_funcs):
            return CompetitiveLearning.competitive_learning(dataset, eta, num_clusters, iterations, score_funcs)

        run_function.params = params
        run_function.alg_name = "Competitive Learning"
        return run_function

    def build_pso_function(self, num_particles, num_centroids, inertia, accel_1, accel_2, max_iter):
        params = locals()

        def run_function(dataset, score_funcs):
            return PSO.pso(num_particles, num_centroids, inertia, accel_1, accel_2, max_iter, dataset, score_funcs=score_funcs)

        run_function.params = params
        run_function.alg_name = "PSO"
        return run_function

if __name__ == '__main__':
    # Analyze.analyze(dataset, "test", 10, build_kMeans_func(2), score_list)
    root = Tk()
    app = build_GA_Menu(root)
    root.mainloop()

    # data1 = [(random.uniform(0, 1), random.uniform(0, 1)) for i in range(50)]
    # data2 = [(random.uniform(3, 4), random.uniform(4, 5)) for i in range(50)]
    # data = data1 + data2
    #
    # result = PSO.pso(10, 2, 0.75, 0.75, 1.2, 100, data)
    #
    # x = []
    # y = []
    # for pt in data:
    #     x.append(pt[0])
    #     y.append(pt[1])
    #
    # plt.scatter(x, y)
    #
    # r_x = []
    # r_y = []
    # for pt in result:
    #     r_x.append(pt[0])
    #     r_y.append(pt[1])
    #
    # plt.scatter(r_x, r_y, c='red')
    # plt.draw()
    #
    # # plt.figure()
    # # plt.plot(result[1])
    # plt.show()


    # score_list = [score_funcs.score_1, score_funcs.score_2]
    # dataset = [(1,1), (2,2), (10,10), (11,11)]
    # Analyze.analyze(dataset, "test", 10, build_kMeans_func(2), score_list)

    # dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Ex_Clusters', 'globular.txt')
    # print(dir_path)
