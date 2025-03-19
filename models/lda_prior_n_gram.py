import numpy as np
import random
from lda import LDA_Gibbs, load_conversation_data
from smoothed_n_gram import NGramMarkovChain


class LDA_NGramModel:
    def __init__(self, lda_model, ngram_model):
        self.lda_model = lda_model
        self.ngram_model = ngram_model
        self.topic_word_dist = self.lda_model.get_topic_word_distribution()

    def predict_next_word(self, context, doc_id):
        """
        Predicts the next word using a combination of the n-gram Markov model
        and topic probabilities from LDA.
        """
        if isinstance(context, list):
            context = tuple(context)

        topic_probs = self.lda_model.get_document_topic_distribution()[doc_id]

        if context not in self.ngram_model.transitions:
            return random.choice(list(self.ngram_model.vocabulary))

        word_counts = self.ngram_model.transitions[context]
        total_count = sum(word_counts.values())

        word_ids = np.array(
            [
                self.lda_model.word_to_id.get(word, -1)
                for word in self.ngram_model.vocabulary
            ]
        )
        valid_indices = word_ids >= 0
        word_ids = word_ids[valid_indices]

        word_topic_probs = np.dot(topic_probs, self.topic_word_dist[:, word_ids])
        ngram_probs = np.array(
            [
                (word_counts.get(word, 0) + self.ngram_model.alpha)
                / (
                    total_count
                    + self.ngram_model.alpha * len(self.ngram_model.vocabulary)
                )
                for word in self.ngram_model.vocabulary
            ]
        )[valid_indices]

        final_probs = ngram_probs * word_topic_probs

        if final_probs.sum() == 0:
            return random.choice(list(self.ngram_model.vocabulary))

        final_probs /= final_probs.sum()
        return np.random.choice(
            np.array(list(self.ngram_model.vocabulary))[valid_indices], p=final_probs
        )

    def generate_sentence(self, start_words, doc_id, max_length=25):
        """Generates a sentence using LDA-weighted n-gram predictions."""
        if not self.ngram_model.transitions:
            return "Model not trained yet."

        sentence = list(start_words)

        for _ in range(max_length - len(sentence)):
            context = tuple(sentence[-self.ngram_model.n :])
            next_word = self.predict_next_word(context, doc_id)

            if next_word is None:
                break

            sentence.append(next_word)

        return " ".join(sentence)


if __name__ == "__main__":
    file_path = "conversation_datasets/cleaned_files/cleaned_eli5_entries.txt"
    documents = load_conversation_data(file_path)

    lda_model = LDA_Gibbs(n_topics=5, alpha=0.1, beta=0.1, n_iter=1)
    lda_model.fit(documents)

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ngram_model = NGramMarkovChain(n=3, alpha=0.01)
    ngram_model.train(text_data)

    combined_model = LDA_NGramModel(lda_model, ngram_model)

    start_words = ["hello", "how", "are"]
    generated_sentence = combined_model.generate_sentence(start_words, doc_id=0)
    print("Generated Sentence:", generated_sentence, "\n")

    generated_sentence = combined_model.generate_sentence(start_words, doc_id=0)
    print("Generated Sentence:", generated_sentence, "\n")

    generated_sentence = combined_model.generate_sentence(start_words, doc_id=0)
    print("Generated Sentence:", generated_sentence, "\n")
