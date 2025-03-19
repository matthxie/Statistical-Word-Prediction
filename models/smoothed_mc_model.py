import math
import numpy as np
import re


class SmoothedNGramMCModel:
    def __init__(self):
        pass

    def get_word_probability(self, context, next_word):
        """
        Calculate the probability of a word given a context using Laplace smoothing.

        Args:
            context: Tuple of context words
            next_word: Word to calculate probability for

        Returns:
            float: Probability of the next word given the context
        """
        if context in self.transitions:
            word_counts = self.transitions[context]
            total_count = sum(word_counts.values())
            count = word_counts.get(next_word, 0)

            # Apply Laplace formula: (count + alpha) / (total_count + alpha * |V|)
            return (count + self.alpha) / (
                total_count + self.alpha * len(self.vocabulary)
            )
        else:
            # Uniform probability for unseen contexts
            return 1.0 / len(self.vocabulary) if self.vocabulary else 0.0

    def calculate_surprisal(self, context, next_word):
        """
        Calculate surprisal (-log2(probability)) for a word given a context.

        Args:
            context: Tuple of context words
            next_word: Word to calculate surprisal for

        Returns:
            float: Surprisal value in bits
        """
        prob = self.get_word_probability(context, next_word)
        return -math.log2(prob) if prob > 0 else float("inf")

    def calculate_sentence_metrics(self, sentence):
        """
        Calculate surprisal, entropy, and perplexity for a given sentence.
        """
        if isinstance(sentence, str):
            words = re.findall(r"\b\w+\b", sentence.lower())
        else:
            words = sentence

        if len(words) < self.n:
            return {
                "warning": f"Sentence too short for {self.n}-gram analysis",
                "surprisals": [],
                "entropy": float("nan"),
                "perplexity": float("nan"),
            }

        surprisals = []
        word_probs = []

        # For each position where we have a full context
        for i in range(len(words) - (self.n)):
            context = tuple(words[i : i + self.n])
            next_word = words[i + self.n]

            # Calculate probability and surprisal
            prob = self.get_word_probability(context, next_word)
            surprisal = -math.log2(prob) if prob > 0 else float("inf")

            surprisals.append(surprisal)
            word_probs.append(prob)

            # Debug output - uncomment to see what's happening
            # print(f"Context: {context}, Next word: {next_word}, Prob: {prob}, Surprisal: {surprisal}")

        # Calculate entropy and perplexity
        if surprisals:
            # Filter out infinite values for mean calculation
            finite_surprisals = [s for s in surprisals if s != float("inf")]
            if finite_surprisals:
                entropy = np.mean(finite_surprisals)
                perplexity = 2**entropy
            else:
                entropy = float("inf")
                perplexity = float("inf")
        else:
            entropy = float("nan")
            perplexity = float("nan")

        return {
            "surprisals": surprisals,
            "word_probabilities": word_probs,
            "entropy": entropy,
            "perplexity": perplexity,
        }

    def evaluate_text(self, text, verbose=False):
        """
        Evaluate a multi-sentence text.

        Args:
            text: String containing multiple sentences to evaluate
            verbose: Whether to print detailed metrics

        Returns:
            dict: Dictionary containing average metrics
        """
        # Split text into sentences (simple split by period for demonstration)
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        all_surprisals = []
        all_entropies = []
        all_perplexities = []

        for i, sentence in enumerate(sentences):
            metrics = self.calculate_sentence_metrics(sentence)

            if "warning" not in metrics:
                all_surprisals.extend(metrics["surprisals"])

                if not math.isnan(metrics["entropy"]) and metrics["entropy"] != float(
                    "inf"
                ):
                    all_entropies.append(metrics["entropy"])

                if not math.isnan(metrics["perplexity"]) and metrics[
                    "perplexity"
                ] != float("inf"):
                    all_perplexities.append(metrics["perplexity"])

                if verbose:
                    print(f"\nSentence {i+1}: '{sentence}'")
                    print(
                        f"  Average Surprisal: {np.mean(metrics['surprisals']):.4f} bits"
                    )
                    print(f"  Entropy: {metrics['entropy']:.4f} bits")
                    print(f"  Perplexity: {metrics['perplexity']:.4f}")

        # Calculate averages
        avg_metrics = {
            "avg_surprisal": (
                np.mean(all_surprisals) if all_surprisals else float("nan")
            ),
            "avg_entropy": np.mean(all_entropies) if all_entropies else float("nan"),
            "avg_perplexity": (
                np.mean(all_perplexities) if all_perplexities else float("nan")
            ),
        }

        if verbose:
            print("\nOverall Metrics:")
            print(f"  Average Surprisal: {avg_metrics['avg_surprisal']:.4f} bits")
            print(f"  Average Entropy: {avg_metrics['avg_entropy']:.4f} bits")
            print(f"  Average Perplexity: {avg_metrics['avg_perplexity']:.4f}")

        return avg_metrics
