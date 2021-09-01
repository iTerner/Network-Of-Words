from sklearn import cluster
import numpy as np
from fitter import Fitter, get_common_distributions, get_distributions
import csv
import pandas as pd
import networkx as nx
import community
import matplotlib.pyplot as plt

plt.style.use("ggplot")

LANG_MAP = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}


def convert_to_list(s: str) -> list:
    if s == "[]":
        return []
    return s.strip('][').replace("'", "").split(', ')


def prepare_data(lang: str, vector_size: int, window_size: int) -> pd.DataFrame:
    """
    The function preparing the data for the algorithm
    @param: lang - string representation of the language
    @param: vector_size - the number of dimensions of each word vector
    @param: window_size - the selected window size of the graph
    """
    d1 = pd.read_csv(
        f"graphs/graph_{lang}_wiki_word2vec_{vector_size}_{window_size}_after_filter_1200_w.csv", encoding="latin1")
    d2 = pd.read_csv(
        f"graphs/graph_{lang}_wiki_word2vec_{vector_size}_{window_size}_after_filter_1200.csv", encoding="latin1")
    data = pd.DataFrame()
    data[["word id", "word", "degree", "neighbors-word", "times"]
         ] = d1[["word id", "word", "degree", "neighbors", "times"]]
    data["neighbors"] = d2["neighbors"]
    print(data.head(), data.columns)
    return data


def df_to_graph(data: pd.DataFrame) -> nx.Graph:
    """
    The function convert the dataframe object to a networkx graph object
    @param: data - the dataframe object that have all the information about the graph
    """
    G = nx.Graph()
    # add the nodes to the graph
    for node in data["word id"].values:
        G.add_node(int(node))

    # add the edges to the graph
    index = 0
    try:
        for row in data.values:
            source = int(row[0])
            for target in convert_to_list(row[5]):
                target = int(target)
                if source < target:
                    G.add_edge(source, target)
            index += 1
    except:
        print(row[5])
        print(index)
        exit()

    print(f"number of nodes in {G.number_of_nodes()}")
    print(f"number of edges in {G.number_of_edges()}")
    return G


def compute_features(G: nx.Graph, lang: str, data: pd.DataFrame) -> None:
    """
    The function compute the graph's features.
    @param: G - the graph as a networkx object
    @param: lang - string representation of the language
    @param: data - the dataframe object that have all the information about the graph
    """
    words = data["word"].values
    degrees = data["degree"].values

    # clustering the graph
    print("start clustering the graph")
    partition = community.best_partition(G)
    n = max(partition.values())
    print("number of clusters in out graph", n)
    print("saving the data")
    with open(f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-clustering-labels.csv", "w", encoding='latin-1', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "word", "degree"])
        index = 0
        for key, val in partition.items():
            writer.writerow([key, val, words[index], degrees[index]])

    # clustering information
    df = pd.read_csv(f"{lang}-clustering-labels.csv", encoding="latin1")
    print("saving all the information from the clustering")
    n = max(df["label"].values)

    f = open(
        f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-clustering-information.txt", "w")
    f.write(f"The number of clusters in the graph is {n}\n")

    points = []
    for i in range(n):
        new_data = df[df["label"] == i]
        points.append([i, new_data.shape[0]])

    points.sort(key=lambda x: x[1], reverse=True)
    for p in points:
        f.write(f"The number of words in cluster number {p[0]} is {p[1]}\n")

    # centrality
    print("start computing the centrality of the graph")
    d = nx.algorithms.centrality.degree_centrality(G)
    sorted(d.items(), key=lambda x: x[1], reverse=True)
    print("saving results")
    with open(f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-centrality-results.csv", "w", newline="", encoding="latin-1") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "centrality", "word", "degree"])
        for key, val in sorted(d.items()):
            writer.writerow([key, val, words[key], degrees[key]])

    # betweenness
    print("starting compute the betwennes")
    b = nx.algorithms.centrality.betweenness_centrality(G)
    sorted(b.items(), key=lambda x: x[1], reverse=True)
    print("saving betweenness results")
    with open(f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-betweenness.csv", "w", encoding='latin-1', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "betweenness", "word", "degree"])
        for key, val in b.items():
            writer.writerow([key, val, words[key], degrees[key]])
    print("done!")


def distribution_detection_preparation() -> None:
    """
    The function prepare the data for the distribution detector
    """
    vector_size = int(input("Enter vector size: "))
    window_size = int(input("Enter window size: "))
    for key in LANG_MAP.keys():
        print(
            f"you chose language {key}, vector size {vector_size} and window size {window_size}")
        df = pd.read_csv(
            f"graph_{key}_wiki_word2vec_{vector_size}_{window_size}_after_filter_1200_w.csv", encoding="latin1")
        degrees = df["degree"].values
        distribution_detector(degrees, key)


def save_distribution_detection_results(summary: pd.DataFrame, best: dict, second: dict, lang: str) -> None:
    """
    The function saves the results of the distribution detection process to a csv file.
    @param: summary - the results of each distribution
    @param: best - the best distribution and their parameters.
    @param: second - the second best distribution and their parameters.
    @param: lang - the language
    """
    with open(f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-distribution-detection-results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distribution", *summary.columns])
        for index, row in zip(summary.index, summary.values):
            writer.writerow([index, *row])

        writer.writerow([])
        writer.writerow(["best distribution", "1st parameter",
                         "2nd parameter", "location", "scale"])
        dist = list(best.keys())[0]
        writer.writerow([dist, best[dist]])
        writer.writerow([])
        writer.writerow(["second best distribution", "1st parameter",
                         "2nd parameter", "location", "scale"])
        dist = list(second.keys())[0]
        writer.writerow([dist, *list(second[dist])])


def distribution_detector(degrees: np.array, lang: str) -> None:
    """
    The function detect the degree distribution of a given graph and returns the corresponding 
    distribution parameters
    @param: degrees - the degrees array of the graph
    @param: lang - the language of graph
    """
    print(f"start fitting the distribution of the {LANG_MAP[lang]} graph")
    f = Fitter(degrees, distributions=[
               "gamma", "beta", "chi", "skewnorm", "logistic"])
    # f = Fitter(degrees)
    f.fit()
    df = f.summary()
    plt.xlabel("degree")
    plt.ylabel("frequency")
    plt.title(f"distribution detection for the {LANG_MAP[lang]} graph")
    plt.savefig(f"images/{LANG_MAP[lang]}-distribution-detection.png")
    # plt.show()
    print(f.summary())
    best = f.get_best(method='sumsquare_error')
    second = {}
    second[list(df.index)[1]] = f.fitted_param[list(df.index)[1]]
    print(best)
    print(second)
    save_distribution_detection_results(df, best, lang)


def degree_distribution_plot(graphs: list, langs: list) -> None:
    """
    The function ploting the degree distribution of a given networkx graph.
    @param: graphs - list of graphs, a networkx graph object
    @param: lang - a list of the string representation of the language
    """
    for i, G in enumerate(graphs):
        degrees = [G.degree(n) for n in G.nodes()]
        degrees.sort(reverse=True)
        x = range(len(degrees))
        l = LANG_MAP[langs[i]]
        plt.scatter(x, degrees, s=5, label=l)

    plt.title("degree distribution")
    plt.ylabel("degree")
    plt.legend()
    plt.show()
    plt.savefig(f"images/degree-distribution.png")


def get_graph(lang: str, vector_size: int, window_size: int) -> tuple:
    """
    The function returns the graph of a given file.
    @param: lang - string representation of the language
    @param: vector_size - the vector size of the graph
    @param: window_size - the window size of the graph
    """
    # convert the graph file to dataframe object
    data = prepare_data(lang, vector_size, window_size)
    # convert the dataframe to a graph
    G = df_to_graph(data)

    return G, data


def feature_preparation() -> None:
    """
    The function managing the features computation process
    """
    lang = input("Enter language: ")
    vector_size = int(input("Enter vector size: "))
    window_size = int(input("Enter window size: "))
    print(
        f"you chose language {LANG_MAP[lang]}, vector size {vector_size} and window size {window_size}")

    # get the graph
    G, data = get_graph(lang, vector_size, window_size)

    # comnpute the features of the graph
    compute_features(G, lang, data)

    print("finished!")


def distribution_preparation() -> None:
    """
    The function managing the distribution plot process
    """
    vs, ws, langs = [], [], []
    count = int(input("Enter number of graphs: "))
    for _ in range(count):
        lang = input("Enter language: ")
        vector_size = int(input("Enter vector size: "))
        window_size = int(input("Enter window size: "))
        langs.append(lang)
        vs.append(vector_size)
        ws.append(window_size)

    graphs = []
    for vsize, wsize, l in zip(vs, ws, langs):
        print(
            f"you chose language {LANG_MAP[l]}, vector size {vsize} and window size {wsize}")
        G, data = get_graph(l, vsize, wsize)
        graphs.append(G)

    # plot a histogram of the degree distribution
    degree_distribution_plot(graphs, langs)


def avergae_path_length() -> None:
    """
    The function managing the average shortest path length computation process
    """
    lang = input("Enter language: ")
    vector_size = int(input("Enter vector size: "))
    window_size = int(input("Enter window size: "))
    print(
        f"you chose language {LANG_MAP[lang]}, vector size {vector_size} and window size {window_size}")

    # get the graph
    G, data = get_graph(lang, vector_size, window_size)

    with open(f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-shortest-path.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        print("starting")
        writer.writerow(
            ["number of nodes", "number of edges", "average length"])
        for g in nx.connected_component_subgraphs(G):
            nodes, edges = g.number_of_nodes(), g.number_of_edges()
            print(
                f"computing average path of the subgraph with {nodes} nodes and {edges} edges")
            avg_path = nx.average_shortest_path_length(g)
            print(f"The average path length is {avg_path}")
            writer.writerow([nodes, edges, avg_path])

    print("finished!")


def save_cluster_results(n: int, lang: str, df: pd.DataFrame) -> None:
    """
    The function saves the results of clustering and creates a file with some information about
    the clustering.
    @param: n - the number of clusters
    @param: lang - the language
    @param: df - the data
    """
    print("start saving")
    df.sort_values(by=["label"])
    with open(f"reports/{LANG_MAP[lang]}/cluster-results-{LANG_MAP[lang]}_{n}.csv", "w", newline="", encoding="latin-1") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "word", "label", "degree"])
        for index, row in enumerate(df.values):
            writer.writerow([index, *row])

    g = open(
        f"reports/{LANG_MAP[lang]}/{LANG_MAP[lang]}-clustering-information-{n}.txt", "w")
    g.write(f"The number of clusters in the graph is {n}\n")

    points = []
    for i in range(n):
        new_data = df[df["label"] == i]
        points.append([i, new_data.shape[0]])

    points.sort(key=lambda x: x[1], reverse=True)
    for p in points:
        g.write(f"The number of words in cluster number {p[0]} is {p[1]}\n")

    print(f"Saved {LANG_MAP[lang]} cluster results with {n} clusters")


def graph_to_edge_matrix(G: nx.Graph) -> np.array:
    """
    Convert a networkx graph into an edge matrix.
    @param: G - networkx graph
    """
    print("creating the edge matrix")
    # Initialize edge matrix with zeros
    edge_mat = np.zeros((len(G), len(G)), dtype=int)

    # Loop to set 0 or 1 (diagonal elements are set to 1)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node][neighbor] = 1
        edge_mat[node][node] = 1

    print("edge matrix created successfully!")
    return edge_mat


def advanced_clustering_preparation() -> None:
    """
    The function prepare the data for the advanced clustering process.
    """
    lang = input("Enter language: ")
    vector_size = int(input("Enter vector size: "))
    window_size = int(input("Enter window size: "))
    n = int(input("Enter the number of clusters: "))
    print(
        f"you chose language {LANG_MAP[lang]}, vector size {vector_size} and window size {window_size}")

    df = prepare_data(lang, vector_size, window_size)

    tmp = pd.read_csv(
        f"graphs/graph_{lang}_wiki_word2vec_{vector_size}_{window_size}_after_filter_1200_w.csv", encoding="latin1")
    words, degrees = tmp["word"].values, tmp["degree"].values
    G = df_to_graph(df)
    advanced_clustering(G, words, degrees, lang, n)


def advanced_clustering(G: nx.graph, words: np.array, degrees: np.array, lang: str, n: int) -> None:
    """
    The function managing the advanced clustering process
    @param: G - the graph object
    @param: words - array of the words
    @param: degrees - array of the degrees
    @param: lang - string representation of the language
    @param: n - the number of clusters
    """
    # create the edge matrix of the given graph
    edge_mat = graph_to_edge_matrix(G)

    print(f"starting preform kmeans with {n} clusters")

    kmeans = cluster.KMeans(n_clusters=n)
    # train the model
    kmeans.fit(edge_mat)

    labels = kmeans.labels_
    print(labels)

    df = pd.DataFrame({"word": words, "label": labels, "degree": degrees})

    # save the results
    save_cluster_results(n, lang, df)


def clustering_coefficient(lang: str, G: nx.Graph) -> None:
    """
    The function compute and save the clustering coefficient of the given graph.
    @param: lang - the selected language.
    @param: G - the graph object.
    """
    print("start computing")
    c = nx.average_clustering(G)
    print(f"clsering coefficient is {c}")
    f = open(f"reports/{LANG_MAP[lang]}/clustering-coefficient.txt", "w")
    f.write(f"{LANG_MAP[lang]} clustering coefficient is {c}")


def clustering_coefficient_preparation() -> None:
    """
    The function prepare the data for the clustering coefficient computation
    """
    lang = input("Enter language: ")
    vector_size = int(input("Enter vector size: "))
    window_size = int(input("Enter window size: "))
    print(
        f"you chose language {LANG_MAP[lang]}, vector size {vector_size} and window size {window_size}")
    df = prepare_data(lang, vector_size, window_size)

    G = df_to_graph(df)
    clustering_coefficient(lang, G)


def centrality_plot() -> None:
    """
    The function create the centrality histogram plot
    """
    lang = input("Enter language: ")
    data = pd.read_csv(
        f"reports/{LANG_MAP[lang]}/{lang}-centrality-results.csv", encoding="latin-1")
    plt.hist(data["centrality"], bins=len(data.centrality.unique()),
             label=LANG_MAP[lang])

    plt.xlabel("Centrality value".upper())
    plt.ylabel("frequency".upper())
    plt.title(f"{LANG_MAP[lang]} Centrality results".upper())
    plt.savefig(f"images/{LANG_MAP[lang]}-centrality.png")


if __name__ == "__main__":
    o = int(input(
        "Enter 1 to compute graph feature\n2 for graph distribution\n3 for average path length\n4 for advanced clustering\n5 for distribution detection\n6 for clustering coefficient\n7 for centrality plot: "))
    if o == 1:
        feature_preparation()
    elif o == 2:
        distribution_preparation()
    elif o == 3:
        avergae_path_length()
    elif o == 4:
        advanced_clustering_preparation()
    elif o == 5:
        distribution_detection_preparation()
    elif o == 6:
        clustering_coefficient_preparation()
    elif o == 7:
        centrality_plot()
