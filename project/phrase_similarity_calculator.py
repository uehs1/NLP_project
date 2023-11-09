import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import logging

class PhraseSimilarityCalculator:
    def __init__(self, word2vec_loader, phrases_df):
        self.wv = word2vec_loader.wv
        self.phrases_df = phrases_df

    def calculate_similarity(self, phrase1, phrase2):
        try:
            tokens1 = phrase1.split()
            tokens2 = phrase2.split()

            embeddings1 = [self.wv[token] for token in tokens1 if token in self.wv.vocab]
            embeddings2 = [self.wv[token] for token in tokens2 if token in self.wv.vocab]

            if not embeddings1 or not embeddings2:
                return None

            phrase_vector1 = normalize(sum(embeddings1).reshape(1, -1))
            phrase_vector2 = normalize(sum(embeddings2).reshape(1, -1))

            similarity = 1 - cosine(phrase_vector1, phrase_vector2)
            return similarity

        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            raise

    def batch_execution(self):
        similarity_results = []

        for i in range(len(self.phrases_df)):
            for j in range(i + 1, len(self.phrases_df)):
                phrase1 = self.phrases_df['Phrase'][i]
                phrase2 = self.phrases_df['Phrase'][j]

                similarity = self.calculate_similarity(phrase1, phrase2)

                if similarity is not None:
                    similarity_results.append((phrase1, phrase2, similarity))

        return pandas.DataFrame(similarity_results, columns=['Phrase1', 'Phrase2', 'Similarity'])

    def on_the_fly_execution(self, input_phrase):
        similarities = []

        for i in range(len(self.phrases_df)):
            phrase = self.phrases_df['Phrase'][i]
            similarity = self.calculate_similarity(input_phrase, phrase)

            if similarity is not None:
                similarities.append((phrase, similarity))

        closest_match = max(similarities, key=lambda x: x[1])
        return closest_match
