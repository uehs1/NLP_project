from word2vec_loader import Word2VecLoader
from phrase_similarity_calculator import PhraseSimilarityCalculator
import logging
import pandas as pd
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main():
    setup_logging()

    try:
        # Replace 'path_to_downloaded_file' and 'path_to_phrases_csv' with actual paths
        word2vec_loader = Word2VecLoader('C:/Users/PALAK AGARWAL/Desktop/NLP_Project/project/GoogleNews-vectors-negative300.bin')
        phrases_df = pd.read_csv('C:/Users/PALAK AGARWAL/Desktop/NLP_Project/project/phrases.csv')

        phrase_similarity_calculator = PhraseSimilarityCalculator(word2vec_loader, phrases_df)

        # Batch Execution
        batch_similarity_df = phrase_similarity_calculator.batch_execution()
        batch_similarity_df.to_csv('batch_similarity_results.csv', index=False)

        # On-the-fly Execution
        user_input = "Your user-input phrase here"
        on_the_fly_result = phrase_similarity_calculator.on_the_fly_execution(user_input)
        print(f"On-the-fly closest match: {on_the_fly_result[0]}, Similarity: {on_the_fly_result[1]}")

    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
