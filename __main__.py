import Analyze
import score_funcs
import KMeans
import DBSCAN
import os
import random
from tkinter import *
from tkinter import ttk


class build_GA_Menu(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_gui()

    def init_gui(self):
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
        pass

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

if __name__ == '__main__':
    root = Tk()
    app = build_GA_Menu(root)
    root.mainloop()

    # score_list = [score_funcs.score_1, score_funcs.score_2]
    # dataset = [(1,1), (2,2), (10,10), (11,11)]
    # Analyze.analyze(dataset, "test", 10, build_kMeans_func(2), score_list)

    # dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Ex_Clusters', 'globular.txt')
    # print(dir_path)
