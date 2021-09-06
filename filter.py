"""
This script is responsible for filtering the embedding file to the selected threshold and saving the results.
"""

import pandas as pd
import csv


def data_preparation() -> tuple:
    langs, sizes, vec_sizes, win_sizes = [], [], [], []
    lang = input("Enter language: ")
    size = int(input("Enter threshold: "))
    vec_size = int(input("Enter vector size: "))
    win_size = int(input("Enter window size: "))
    while lang != "0" and size != 0 and vec_size != 0 and win_size != 0:
        langs.append(lang)
        sizes.append(size)
        vec_sizes.append(vec_size)
        win_sizes.append(win_size)
        lang = input("Enter language: ")
        size = int(input("Enter threshold: "))
        vec_size = int(input("Enter vector size: "))
        win_size = int(input("Enter window size: "))
    return langs, sizes, vec_size, win_size


def data_filter(lang: str, size: int, vector_size: int, win_size: int) -> None:
    """
    The function preform the filtering process for the given file
    @param: lang - the string representation of the language
    @param: size - the threshold
    @param: vector_size - the size of the vector of the file
    @param: win_size - the size of the window of the file
    """
    # open the frequency list of the selected language
    freq = pd.read_csv(
        f"dictionaries/{lang}_dictionary.txt", sep="\t", encoding="latin-1")
    freq = freq[freq["times"] >= size]
    words = freq["word"].values
    print(words)
    print(words.shape)
    FILENAME = str(f"{lang}_wiki_word2vec_") + \
        str(vector_size) + str("_") + str(win_size)
    data = pd.read_csv(f"{FILENAME}.txt", sep=" ",
                       skiprows=1, encoding="latin-1")
    index = 0
    total = len(words)
    print(
        f"starting with threshold {size} and vector size {vector_size} and window size {win_size}")

    # start the filtering process
    with open(f"embeddings/{FILENAME}_after_filter_{size}.txt", "w", newline="", encoding="latin-1") as f:
        writer = csv.writer(f)
        cols = data.columns
        for word in words:
            row = data[data[cols[0]] == word].values
            if len(row) > 0:
                writer.writerow(row[0])
            else:
                print(word)

            index += 1
            if index % 1000 == 0:
                print(f"finished {index} words, {index / total * 100}%")

    dt = pd.read_csv(f"{FILENAME}_after_filter_{size}.txt")
    print(dt.head())
    print(dt.shape)


def main() -> None:
    """
    The function managing the filtering process.
    """
    # get all the data
    langs, sizes, vec_sizes, win_sizes = data_preparation()
    # preform the filter process for each file
    for lang, size, vector_size, win_size in zip(langs, sizes, vec_sizes, win_sizes):
        data_filter(lang, size, vector_size, win_size)
    print("finished!")


if __name__ == "__main__":
    main()
