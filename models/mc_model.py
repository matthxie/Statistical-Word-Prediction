class NGramMCModel:
    def __init__(self):
        pass

    def calculate_sentence_metrics(self, sentence):
        """
        Calculate surprisal, entropy, and perplexity for a sentence.

        Args:
            sentence: Input sentence to evaluate

        Returns:
            tuple: (average_surprisal, entropy, perplexity)
        """
        import math
        import re

        # The issue is likely in the n-gram size; the debug output shows 8-grams
        # but we should use the model's configured n value
        n = self.n
        words = re.findall(r"\b\w+\b", sentence.lower())

        if len(words) <= n:
            print(f"Warning: Sentence is too short for {n}-gram analysis")
            return None, None, None

        total_log_prob = 0.0
        word_count = 0

        # Loop through each position where we can predict the next word
        for i in range(len(words) - n):
            context = tuple(words[i : i + n])
            next_word = words[i + n]

            # Check if we've seen this context before
            if context in self.transitions:
                # Get all possible next words for this context
                possible_next_words = self.transitions[context]
                total_transitions = len(possible_next_words)

                # Count occurrences of the specific next word
                next_word_count = possible_next_words.count(next_word)

                # Calculate probability with smoothing
                if next_word_count > 0:
                    probability = next_word_count / total_transitions
                else:
                    # Apply Laplace smoothing
                    # Get vocabulary size from all observed next words
                    all_next_words = set()
                    for words_list in self.transitions.values():
                        all_next_words.update(words_list)
                    vocab_size = len(all_next_words) or 1  # Avoid division by zero

                    # Add 1 to count and add vocab size to denominator
                    probability = 1 / (total_transitions + vocab_size)
            else:
                # If context not seen, use uniform distribution over vocabulary
                all_next_words = set()
                for words_list in self.transitions.values():
                    all_next_words.update(words_list)
                vocab_size = len(all_next_words) or 1  # Avoid division by zero
                probability = 1 / vocab_size

            # Ensure minimum probability to avoid log(0)
            probability = max(probability, 1e-10)

            # Calculate log probability (base 2 for bits)
            log_prob = math.log2(probability)
            total_log_prob += log_prob
            word_count += 1

        if word_count == 0:
            print(f"Warning: No valid n-grams found in sentence")
            return None, None, None

        # Average negative log probability = surprisal
        average_surprisal = -total_log_prob / word_count

        # For n-gram models, entropy equals average surprisal in this calculation
        entropy = average_surprisal

        # Perplexity is 2^entropy
        perplexity = 2**entropy

        return average_surprisal, entropy, perplexity

    def evaluate_corpus(self, text):
        """
        Evaluate a corpus of text containing multiple sentences.

        Args:
            text: Text containing multiple sentences separated by periods

        Returns:
            dict: Dictionary containing metrics for each sentence and overall averages
        """
        # Split text into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        results = {
            "sentences": [],
            "overall": {
                "average_surprisal": 0.0,
                "average_entropy": 0.0,
                "average_perplexity": 0.0,
            },
        }

        valid_sentence_count = 0
        total_surprisal = 0.0
        total_entropy = 0.0
        total_perplexity = 0.0

        # Process each sentence
        for i, sentence in enumerate(sentences):

            surprisal, entropy, perplexity = self.calculate_sentence_metrics(sentence)

            if surprisal is not None:  # Only count valid sentences
                sentence_result = {
                    "text": sentence,
                    "average_surprisal": surprisal,
                    "entropy": entropy,
                    "perplexity": perplexity,
                }

                results["sentences"].append(sentence_result)

                total_surprisal += surprisal
                total_entropy += entropy
                total_perplexity += perplexity
                valid_sentence_count += 1

        # Calculate overall metrics
        if valid_sentence_count > 0:
            results["overall"]["average_surprisal"] = (
                total_surprisal / valid_sentence_count
            )
            results["overall"]["average_entropy"] = total_entropy / valid_sentence_count
            results["overall"]["average_perplexity"] = (
                total_perplexity / valid_sentence_count
            )

        return results

    def evaluate_text(self, text):
        """
        Evaluate and print results for a text corpus.

        Args:
            text: Text containing multiple sentences separated by periods
        """
        # First, verify the model has been trained
        if not self.transitions:
            print(
                "Error: Model has not been trained. Please train the model before evaluation."
            )
            return

        self.train(text)

        results = self.evaluate_corpus(text)

        print("\n----- Per-Sentence Metrics -----\n")
        for i, sentence in enumerate(results["sentences"]):
            print(f"Sentence {i+1}: '{sentence['text']}'")
            print(f"  Average Surprisal: {sentence['average_surprisal']:.4f} bits")
            print(f"  Entropy: {sentence['entropy']:.4f} bits")
            print(f"  Perplexity: {sentence['perplexity']:.4f}")

        print("\n----- Overall Metrics -----\n")
        print(
            f"  Average Surprisal: {results['overall']['average_surprisal']:.4f} bits"
        )
        print(f"  Average Entropy: {results['overall']['average_entropy']:.4f} bits")
        print(f"  Average Perplexity: {results['overall']['average_perplexity']:.4f}")
