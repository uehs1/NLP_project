import gensim
import pandas as pd
from gensim.models import KeyedVectors
import logging

class Word2VecLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.wv = self.load_word2vec_vectors()

    def load_word2vec_vectors(self):
        try:
            logging.info(f"Loading Word2Vec vectors from: {self.file_path}")
            # Convert the Path object to a string
            file_path_str = str(self.file_path)
            wv = KeyedVectors.load_word2vec_format(file_path_str, binary=True)
            return wv
        except Exception as e:
            logging.error(f"Error loading Word2Vec vectors: {e}")
            raise

    def save_as_flat_file(self, flat_file_path):
        try:
            logging.info(f"Saving Word2Vec vectors as flat file: {flat_file_path}")
            self.wv.save_word2vec_format(flat_file_path)
        except Exception as e:
            logging.error(f"Error saving Word2Vec vectors as flat file: {e}")
            raise
