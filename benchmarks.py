import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)


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


if __name__ == "__main__":
    from models.smoothed_n_gram import SmoothedNGramMC
    from models.n_gram_markov_chain import NGramMC

    datasets = [
        "conversation_datasets/cleaned_files/cleaned_dailydialog.txt",
        "conversation_datasets/cleaned_files/cleaned_human_chat.txt",
        "conversation_datasets/cleaned_files/cleaned_eli5_entries.txt",
    ]

    smoothed_mc = SmoothedNGramMC(n=3, alpha=0.000001)
    n_gram_mc = NGramMC(n=3)

    smoothed_mc.train_on_files(datasets)
    n_gram_mc.train_on_files(datasets)

    smoothed_mc_sentences = ""
    n_gram_mc_sentences = ""

    for i in range(5):
        smoothed_mc_sentences += smoothed_mc.generate_sentence() + "."
        n_gram_mc_sentences += n_gram_mc.generate_sentence() + "."

    smoothed_mc_metrics = smoothed_mc.evaluate_text(smoothed_mc_sentences, verbose=True)
    print(smoothed_mc_metrics)

    print("\n\n\n ====================================== \n\n\n")

    n_gram_mc_metrics = smoothed_mc.evaluate_text(smoothed_mc_sentences, verbose=True)
    print(n_gram_mc_metrics)
