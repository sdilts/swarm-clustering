import pandas as pd
import os
import numpy as np


def to_tuples(df, needed_values):
    # something like:
    # subset = data_set[['data_date', 'data_1', 'data_2']]
    subset = df[needed_values]
    return [tuple(x) for x in subset.values]


def normalize(df):
    return (df-df.mean())/df.std()

# Working
def load_iris():
    path = os.path.realpath("data/iris.data")
    df = pd.read_csv(path, sep=',',names=['sl','sw','pl','pw','class'], dtype={'sl': np.float64, 'sw': np.float64, 'pl': np.float64,'pw': np.float64, 'class': str})
    val_list = ['sl', 'sw', 'pl', 'pw']
    df_vals = df[val_list]
    #df_norm = (df_vals - df_vals.mean()) / (df_vals.max() - df_vals.min())
    # df_norm['class'] = df['class']
    return to_tuples(df_vals, val_list)

# Working
def load_cust_data():
    path = os.path.realpath("data/whole_cust_data.csv")
    val_list = ['Channel',  'Region',  'Fresh',   'Milk',  'Grocery',  'Frozen',  'Detergents_Paper','Delicassen']
    df = pd.read_csv(path,sep=',')
    df_vals = df[val_list]
    return to_tuples(df_vals, val_list)

# Working
def load_glass():
    path = os.path.realpath("data/glass.data")
    val_list = ['ri','na','mg','al','si','k','ca','ba','fe','class']
    df = pd.read_csv(path,sep=',', index_col=0, names=val_list)
    df_vals = df[val_list]
    return to_tuples(df_vals, val_list[:-1])

# Working
def load_seeds():
    path = os.path.realpath("data/seeds_dataset.txt")
    val_list = ['area','perimeter','compactness','k_length','k_width','a_coef','groove_len','class']
    df = pd.read_csv(path, names=val_list, delim_whitespace=True)
    df_vals = df[val_list]
    # df_norm = normalize(df[val_list[:-1]])
    return to_tuples(df_vals, val_list[:-1])


def load_banknote():
    path = os.path.realpath("data/data_banknote_authentication.txt")
    val_list = ['var','skew','curt','entropy','class']
    df = pd.read_csv(path,sep=',',names=val_list)
    df_norm = normalize(df[val_list[:-1]])
    return to_tuples(df_norm, val_list[:-1])
