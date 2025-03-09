from mc import NGramMarkovChain


model = NGramMarkovChain(n=3)

model.train_on_file("train.txt")

sentence = model.generate_sentence(start_words=["hello", "how", "are"])
print(sentence)

print()

sentence = model.generate_sentence()
print(sentence)
print()

sentence = model.generate_sentence()
print(sentence)
print()

sentence = model.generate_sentence()
print(sentence)
print()
