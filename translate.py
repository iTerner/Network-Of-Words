from googletrans import Translator
import pandas as pd
import csv

SOURCE_LANG = "en"
LANG_MAP = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}


def prepare_data() -> tuple:
    """
    The function prepare the data for the translation process.
    """
    file_path = input("Enter file path: ")
    lang = input("Enter language: ")

    df = pd.read_csv(file_path, encoding='latin-1')
    return file_path, lang, df


def saveing_results(path: str, translate: dict, df: pd.DataFrame) -> None:
    """
    The function saves the results in the given csv file.
    @param: path - the original file path
    @param: translate - a dictionary contains the origin word as a key and the translation as the value
    @param: df - the data
    """
    print("saving results")
    with open(path, "w", newline="", encoding="latin-1") as f:
        writer = csv.writer(f)
        writer.writerow([*df.columns, f"{LANG_MAP[SOURCE_LANG]} translation"])
        for row, val in zip(df.values, translate.values()):
            try:
                writer.writerow([*row, val])
            except:
                writer.writerow([*row, "cant load translation"])


def translation() -> None:
    """
    The function manage the translation process.
    """
    path, lang, df = prepare_data()
    translate = {}
    translator = Translator()
    index = 0
    words = df["word"].values
    total = len(words)
    print(
        f"start translating from {LANG_MAP[lang]} to {LANG_MAP[SOURCE_LANG]}")
    print("total words", total)

    for word in words:
        try:
            translation = translator.translate(word, src=lang, dest="en")
            translate[word] = translation.text
            if index % 100 == 0:
                print(f"finished translate {index / total}% of the words")
            index += 1
        except:
            translate[word] = "faild to translate the word"
            index += 1
            print(f"failed to translate {word}")
            # exit()

    # to see the translation dictionary uncomment the following line
    # print(translate)

    # saving the results
    saveing_results(path, translate, df)


if __name__ == "__main__":
    translation()
