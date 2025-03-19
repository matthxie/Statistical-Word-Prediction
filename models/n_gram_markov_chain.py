from models.mc_model import NGramMCModel
import random
import re
from collections import defaultdict


class NGramMC(NGramMCModel):
    def __init__(self, n=2):
        """
        Initialize an n-gram Markov chain model.

        Args:
            n (int): The number of words to use as context for prediction (default: 2)
        """
        self.n = n
        self.transitions = defaultdict(list)
        self.starting_ngrams = []
        super(NGramMC).__init__()

    def _get_ngrams(self, words):
        """Generate n-grams from a list of words."""
        return [tuple(words[i : i + self.n]) for i in range(len(words) - self.n + 1)]

    def train(self, text):
        """Train the Markov chain on the provided text."""
        words = re.findall(r"\b\w+\b", text.lower())

        if len(words) < self.n + 1:
            print(f"Warning: Text too short to train {self.n}-gram model")
            return

        self.starting_ngrams.append(tuple(words[: self.n]))

        ngrams = self._get_ngrams(words)
        for i in range(len(ngrams) - 1):
            current_ngram = ngrams[i]
            next_word = words[i + self.n]
            self.transitions[current_ngram].append(next_word)

    def predict_next_word(self, context):
        """
        Predict the next word given the context (last n words).

        Args:
            context: List or tuple of the last n words
        """
        if isinstance(context, list):
            context = tuple(context)

        if len(context) < self.n:
            print(f"Warning: Context needs {self.n} words")
            return None
        elif len(context) > self.n:
            context = context[-self.n :]

        if context in self.transitions and self.transitions[context]:
            return random.choice(self.transitions[context])
        else:
            if self.n > 1 and len(context) > 1:
                smaller_context = context[1:]
                return self.predict_next_word(smaller_context)
            else:
                if self.starting_ngrams:
                    random_start = random.choice(self.starting_ngrams)
                    return random.choice(
                        self.transitions.get(random_start, [random_start[-1]])
                    )
                return None

    def generate_sentence(self, start_words=None, max_length=25):
        """
        Generate a sentence starting with the given words.

        Args:
            start_words: List of words to start the sentence with
            max_length: Maximum length of the generated sentence
        """
        if not self.transitions:
            return "Model not trained yet."

        if start_words is None or len(start_words) < self.n:
            if not self.starting_ngrams:
                return "Model not trained properly."

            if start_words is None:
                start_ngram = random.choice(self.starting_ngrams)
                sentence = list(start_ngram)
            else:
                sentence = list(start_words)
                while len(sentence) < self.n:
                    random_start = random.choice(self.starting_ngrams)
                    sentence.append(random_start[len(sentence)])
        else:
            if len(start_words) > self.n:
                start_words = start_words[-self.n :]
            sentence = list(start_words)

        for _ in range(max_length - len(sentence)):
            current_context = tuple(sentence[-self.n :])
            next_word = self.predict_next_word(current_context)

            if next_word is None:
                break

            sentence.append(next_word)

        return " ".join(sentence)

    def train_on_files(self, file_paths):
        """Train the model on text from a file."""
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                self.train(text)
                return True
            except Exception as e:
                print(f"Error training on file: {e}")
                return False

    def evaluate_text(self, text, verbose=False):
        return super(NGramMC, self).evaluate_text(text)
        # return super(NGramMC, self).evaluate_metrics(text)


if __name__ == "__main__":
    datasets = [
        "conversation_datasets/cleaned_files/cleaned_dailydialog.txt",
        "conversation_datasets/cleaned_files/cleaned_human_chat.txt",
        "conversation_datasets/cleaned_files/cleaned_eli5_entries.txt",
    ]
    model = NGramMC(n=3)
    model.train_on_files(datasets)

    sentence = model.generate_sentence()
    print(sentence)
