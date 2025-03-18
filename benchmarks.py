import nltk
import math
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

nltk.download("punkt")


def calculate_bleu(reference, prediction):
    reference_tokens = nltk.word_tokenize(reference.lower())
    prediction_tokens = nltk.word_tokenize(prediction.lower())

    score = sentence_bleu([reference_tokens], prediction_tokens)
    return score


def calculate_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores


def calculate_top1_accuracy(reference_file, prediction_file):
    with open(reference_file, "r") as ref_file, open(prediction_file, "r") as pred_file:
        references = ref_file.readlines()
        predictions = pred_file.readlines()

    correct_predictions = 0
    total_predictions = 0

    for ref, pred in zip(references, predictions):
        ref_words = ref.strip().split()
        pred_word = pred.strip().split()[-1]

        if len(ref_words) > 1 and ref_words[-1] == pred_word:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def _get_probabilities(self, context):
        """Calculate P(w | C) for each possible next word."""
        if context not in self.transition_counts:
            return {}

        total_count = sum(self.transition_counts[context].values())
        return {word: count / total_count for word, count in self.transition_counts[context].items()}

def compute_surprisal(self, context, word):
    probabilities = self._get_probabilities(context)
    if word in probabilities and probabilities[word] > 0:
        return -math.log2(probabilities[word])
    return float('inf')  # If the word never appeared, it's infinitely surprising

def compute_entropy(self, context):
    probabilities = self._get_probabilities(context)
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0)

def compute_perplexity(self, test_text):
    words = test_text.lower().split()
    ngrams = self._get_ngrams(words)
    log_prob_sum = 0
    n = len(ngrams)

    for i in range(n - 1):
        context = ngrams[i]
        word = words[i + self.n]
        word_probs = self.transitions[context]

        vocab_size = len(set(word for counts in self.transitions.values() for word in counts))
        total_count = self.total_counts[context] + self.alpha * vocab_size
        prob = (word_probs.get(word, 0) + self.alpha) / total_count

        log_prob_sum += math.log(prob)

    return math.exp(-log_prob_sum / n) if n > 0 else float('inf')  
