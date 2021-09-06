"""
This script is responsible for the creation of the graph from our embedding file. 
"""

import io
import codecs
import numpy as np
from numpy.lib.npyio import save
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time
from scipy.stats import norm
import csv
import pandas as pd


def load_vectors_array_imp(fname: str) -> tuple:
    """
    The function load the words and their vectors of the selected embedding and put them in an array.
    @param: fname - the selected embedding name 
    """
    path = f"embeddings/{fname}"
    fin = io.open(path, 'r', encoding='latin-1',
                  newline='\n', errors='ignore')
    # fin = io.open(fname, 'r', encoding='latin-1', newline='\n')
    s = fin.readline().split(",")
    print(s)
    if len(s) == 2:
        n, d = map(int, s)
    else:
        print("error")
        n, d = map(int, s)
    print(n, d)
    word2id = {}
    words = list()
    A = np.zeros((n, d), dtype=float)
    # B = np.zeros((n,d), dtype=float)
    print(A.shape)
    i = 0
    for line in fin:
        tokens = line.rstrip().split(",")
        # data[tokens[0]] = map(float, tokens[1:])
        word = tokens[0]
        try:
            embedding = np.array([float(val) for val in tokens[1:]])
            assert (embedding.size == d)
        except:
            print(i)
            print(word)
            print(tokens[1:])
            exit()
        words.append(word)
        A[i] = embedding
        # normalized = embedding / np.linalg.norm(embedding)
        # B[i] = normalized
        # print(word)
        # print(i)
        word2id[word.strip()] = i
        i = i+1
#        A = np.append(A, embedding)
#        A.append(embedding)
#        data[tokens[0]] = embedding
    assert(i == n)
    # B = normalize(A, norm='l2')

    # print (A.shape)
    # print (A[0])
    # print (words[0])
#    print (data.())
#    assert(len(data)==n)
    # verify n,d
    return (A, words, word2id)


def data_preparation() -> list:
    """
    The function prepare all the files on which we will run the graph construction algorithm
    """
    f = []
    option = int(input("Enter 1 for file or 2 for manual: "))
    if option == 2:
        filename = input("model: ")
        while filename != "0":
            f.append(filename)
            filename = input("model: ")
    elif option == 1:
        fname = input("Enter file name: ")
        with open(fname, "r") as g:
            lines = g.readlines()
            for line in lines:
                f.append(line)
    else:
        print("you didnt select valid option")
        exit()

    return f


def create_graph(f: str) -> None:
    """
    The function create the graph of the given file.
    @param: f - the file name
    """
    # get the language
    tmp = f.split("_")
    lang = tmp[0]
    print(f"selected language is {lang}")
    # open the frequency file of the selected language
    d = pd.read_csv(
        f"dictionaries/{lang}_dictionary.txt", sep="\t", encoding="latin-1")

    # starting the building process
    index = 0
    print(f"starting with {f}")
    (A, words, word2id) = load_vectors_array_imp(f)
    save_name = f.replace(".txt", "")
    m = np.mean(A, axis=0)
    print("mean:\n")
    print(m)
    if False:  # demean?
        B = A - m
    else:
        B = A
        # B = demean(A, axis=0)
    print("mean demeaned:\n")
    print(np.mean(B, axis=0))

    #    print (B[i])
    C = normalize(B, norm='l2', axis=0)
    # print (C[i])
    print("mean normalized:\n")
    print(np.mean(C, axis=0))

    nstd = 3
    n = C.shape[0]

    node_stats = {}
    with open(f"word-stats/word_stats_{save_name}.csv", 'w', newline='', encoding='latin-1') as csvfile:
        print("calculating word stats")
        writer = csv.writer(csvfile)
        writer.writerow(["index", "word", "min", "mean", "max", "std"])

        print("size:", n)
        for i in range(n):
            dp = np.dot(C, C[i])
            distances = [dp[k] for k in range(n) if k != i]
            (mean, std) = (np.mean(distances), np.std(distances))
            (mn, mx) = (min(distances), max(distances))
            writer.writerow([i, words[i], mn, mean, mx, std])
            node_stats[i] = mean+nstd*std

    #            edges = [words[k] for k in range(n) if k!=i and dp[k]>h["mean"]+3* h["std"]]
    #            edges2 = [words[k] for k in range(n) if k!=i and dp[k]<h["mean"]-3* h["std"]]
    #            print (i, words[i], len(edges), edges, edges2)

    with open(f"graphs/graph_{save_name}.csv", 'w', newline='', encoding='latin-1') as csvfile, open(f"graphs/graph_{save_name}_w.csv", 'w', newline='', encoding='latin-1') as csvfile_w:
        print("building the graph")
        writer = csv.writer(csvfile)
        writer_w = csv.writer(csvfile_w)
        writer.writerow(["word", "degree", "neighbors"])
        writer_w.writerow(["word id", "word", "degree", "neighbors", "times"])

        for i in range(n):
            dp = np.dot(C, C[i])
            edges = [k for k in range(
                n) if k != i and dp[k] > node_stats[i] and dp[k] > node_stats[k]]
            row = [i, len(edges)]
            row.append(edges)
            writer.writerow(row)

            if (len(edges) > 0):
                edges = [words[k] for k in range(
                    n) if k != i and dp[k] > node_stats[i] and dp[k] > node_stats[k]]
                row = [i, words[i], len(edges)]
                row.append(edges)
                freq = 0
                try:
                    new_data = d[d["word"] == words[i]]
                    new_data = new_data.values
                    freq = new_data[0][2]
                except:
                    freq = -1
                row.append(freq)
                writer_w.writerow(row)
            else:
                try:
                    new_data = d[d["word"] == words[i]]
                    new_data = new_data.values
                    freq = new_data[0][2]
                except:
                    freq = -1
                row = [i, words[i], len(edges), edges, freq]
                writer_w.writerow(row)
        index += 1
        print(f"done with {f}")


def main() -> None:
    """
    The functiom managing the graph building process.
    """
    f = data_preparation()
    for filename in f:
        create_graph(filename)

    print("finished!")


if __name__ == '__main__':
    main()


##
    # h = {}
    # (h["mean"], h["std"], h["min"], h["max"], h["index"], h["word"], h["nstd"]) = (None, None, None, None, None, None, None)
    # with open("word_stats.csv", 'w', newline='', encoding='utf-8') as csvfile:
    #     print ("writing word stats")
    #     writer = csv.DictWriter(csvfile, h.keys())
    #     writer.writeheader()
    #
    #     print ("size:", n)
    #     for i in range(n):
    #         dp = np.dot(C, C[i])
    #         distances = [dp[k] for k in range(n) if k!=i]
    #         (h["mean"], h["std"]) = (np.mean(distances), np.std(distances))
    #         (h["min"], h["max"]) = (min(distances), max(distances))
    #         (h["index"], h["word"], h["nstd"]) = (i, words[i], nstd* h["std"])
    #         writer.writerow(h)
    #         node_stats[i] = h["mean"]+nstd*h["std"]
    #         h = {}
