"""
The script is responsible for downloading the corpora from Wikipedia. 
"""

import jieba
import logging
import math
import os
import requests
from gensim.corpora import Dictionary

from gensim.corpora.wikicorpus import WikiCorpus
from tqdm import tqdm


def download_file(url: str, file: str):
    """
    The function download the file.
    @param: url - the url of the file
    @param: file - the file name
    """
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    logging.info('Downloading %s to %s', url, file)
    with open(file, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size / block_size), unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    logging.info('Done')


def download_wiki_dump(lang: str, path: str):
    """
    The function download the corpus of the selected language.
    @param: lang: the selected language.
    @param: path: the path to the corpus
    """
    url = 'https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream.xml.bz2'
    if not os.path.exists(path):
        download_file(url.format(lang=lang), path)
    else:
        num_lines = sum(1 for line in open(url))
        print(num_lines)
        logging.info('%s exists, skip download', path)


class WikiSentences:
    def __init__(self, wiki_dump_path: str, lang: str):
        logging.info('Parsing wiki corpus')
        dict_file = f"data/{lang}_dictionary.txt"
        if os.path.exists(dict_file):
            logging.info("loading dictionary file")
            dictionary = Dictionary.load_from_text(dict_file)
            self.wiki = WikiCorpus(
                wiki_dump_path, dictionary=dictionary, token_min_len=1)
        else:
            self.wiki = WikiCorpus(wiki_dump_path, token_min_len=1)
            logging.info("writing dictionary file")
            self.wiki.dictionary.save_as_text(dict_file, sort_by_word=False)
        self.lang = lang

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            if self.lang == 'zh':
                yield list(jieba.cut(''.join(sentence), cut_all=False))
            else:
                yield list(sentence)
