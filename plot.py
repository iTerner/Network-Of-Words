"""
This script is responsible for creating the average degree and std plots of the graph in addition to the
frequency plots of each degree.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def freq_degree_plot(data: pd.DataFrame, vector_size: int, window_size: int, threshold: int, xloglabel=False, yloglabel=False) -> None:
    """
    The function create a plot of the degree of the graph nodes and the number of time each
    node appear on the embedding, with an option to activate a log to each axis.
    @param: data - the graph data, DataFrame object 
    @param: vector_size - the vector's number of dimensions
    @param: window_size - the window size of the graph
    @param: xloglabel - whether to activate a log to the x axis
    @param: yloglabel - whether to activate a log to the y axis
    """
    plt.clf()
    X = data["times"].values
    Y = data["degree"].values
    x_label, y_label = "number of times", "node degree"
    title = ""
    flag = False
    if xloglabel:
        x_label = "log of number of times"
        X = np.log(X)
        title = "log"
        flag = True
    if yloglabel:
        y_label = "log of node degree"
        Y = np.log(Y)
        if flag:
            title += "_"
        title += "log"
    plt.scatter(X, Y, c="black")
    plt.xlabel(x_label.upper())
    plt.ylabel(y_label.upper())
    plt.title(f"times and degree graph {vector_size}, {window_size}".upper())
    plt.savefig(
        f"{title}_times_degree_{vector_size}_{window_size}_{threshold}.png")
    # plt.show()


def prepare_data(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    The function prepare the data points for the plots
    @param: data - the data we want to plot
    @param: name - the name of the y label feature (in our case, average degree or degree std)
    """
    labels = [1200]
    # labels = data.threshold.unique()
    x, y, l = [], [], []
    for label in labels:
        new_data = data[data["threshold"] == label]
        for size, val in zip(new_data["vector size"].values, new_data[name].values):
            x.append(size)
            y.append(val)
            l.append(label)

    df = pd.DataFrame({"X": x, "Y": y, "L": l})
    return df


def plot(points: pd.DataFrame, title: str, s: int, e: int) -> None:
    """
    Plot and save the figure
    @param: points - the data points
    @param: title - the y label column
    @param: s - the start range of the plot
    @param: e - the end range of the plot
    """
    plt.clf()
    colors = ["red", "black", "blue", "green", "yellow"]
    labels = points.L.unique()
    for i, label in enumerate(labels):
        p = points[points["L"] == label]
        plt.scatter(p["X"], p["Y"], c=colors[i], s=5)

    plt.title(f"RANGE {s} - {e} {title.upper()}")
    plt.xlabel("VECTOR SIZE")
    plt.ylabel(title.upper())
    plt.legend(labels, title="THRESHOLD")
    for i, txt in enumerate(points["X"].values):
        plt.annotate(txt, (points["X"].values[i], points["Y"].values[i]))
    plt.savefig(f"{title}_{s}_{e}")


def freq_plot(vector_size: int, window_size: int, t: str) -> None:
    """
    The function prepare the data for the frequency plots of the graph. The function also apply log to 
    x axis and both the x axis any y axis.
    @param: vector_size - the vector size of the graph
    @param: window_size - the window size of the graph
    @param: t - the title
    """
    filename = f"en_wiki_word2vec_{vector_size}_{window_size}_after_filter_{t}"
    print(f"vector size {vector_size}, window size {window_size}")
    data = pd.read_csv(f"graph_{filename}_w.csv")
    freq_degree_plot(data, vector_size, window_size, t)
    freq_degree_plot(data, vector_size, window_size, t, yloglabel=True)
    freq_degree_plot(data, vector_size, window_size,
                     t, xloglabel=True, yloglabel=True)


def all_graph_plot(s: int, e: int) -> None:
    """
    The function prepare the data for the average degree and standard deviation plots.
    @param: s - the start range of the plot
    @param: e - the end range of the plot
    """
    info = pd.read_csv("feature_results.csv")
    df = info[(info["vector size"] >= s) & (info["vector size"] <= e)]
    titles = ["average degree", "degree std"]
    for title in titles:
        points = prepare_data(df, title)
        plot(points, title, s, e)


def main():
    """
    The function manage the graph parameter selection plots
    """
    i = int(input("Is freq plot needed? enter 1 for yes or 0 for no: "))
    if i != 0:
        features = pd.read_csv("feature_results.csv")
        for vector_size, t in zip(features["vector size"].values, features["threshold"].values):
            freq_plot(vector_size, 5, t)

    start, end = [190], [320]

    for s, e in zip(start, end):
        print(s, e)
        all_graph_plot(s, e)


if __name__ == "__main__":
    main()
