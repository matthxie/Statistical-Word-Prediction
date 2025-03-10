from models.n_gram_markov_chain import NGramMarkovChain
from conversation_datasets.human_chat import HumanChat


model = NGramMarkovChain(n=3)
dataset = HumanChat()

dataset.write_to_file("conversation_datasets/cleaned_files/human_chat.txt")
model.train_on_file("conversation_datasets/cleaned_files/human_chat.txt")

sentence = model.generate_sentence(start_words=["hello", "how", "are"])
print(sentence)

print()

sentence = model.generate_sentence()
print(sentence)
