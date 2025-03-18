import re
import numpy as np
from scipy.sparse import lil_matrix

class LDA_Gibbs:
    def __init__(self, n_topics, alpha, beta, n_iter=1000):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
    
    def fit(self, documents):
        self.documents = documents
        self.n_docs = len(documents)
        self.vocab = list(set(word for doc in documents for word in doc))
        self.n_words = len(self.vocab)
        
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        self.topic_assignments = []
        self.n_wd = lil_matrix((self.n_docs, self.n_words), dtype=int)
        self.n_td = np.zeros((self.n_docs, self.n_topics))
        self.n_t = np.zeros(self.n_topics)
        
        for doc_id, doc in enumerate(documents):
            doc_assignment = []
            for word in doc:
                word_id = self.word_to_id[word]
                topic = np.random.randint(self.n_topics)
                doc_assignment.append(topic)
                
                self.n_wd[doc_id, word_id] += 1
                self.n_td[doc_id, topic] += 1
                self.n_t[topic] += 1
            self.topic_assignments.append(doc_assignment)
        
        for _ in range(self.n_iter):
            self._gibbs_sampling()
    
    def _gibbs_sampling(self):
        for doc_id, doc in enumerate(self.documents):
            for word_idx, word in enumerate(doc):
                word_id = self.word_to_id[word]
                current_topic = self.topic_assignments[doc_id][word_idx]
                
                self.n_wd[doc_id, word_id] -= 1
                self.n_td[doc_id, current_topic] -= 1
                self.n_t[current_topic] -= 1
                
                topic_probs = (self.n_wd[doc_id, word_id] + self.beta) * \
                              (self.n_td[doc_id] + self.alpha) / \
                              (self.n_t + self.n_words * self.beta)
                
                topic_probs /= topic_probs.sum()
                new_topic = np.random.choice(self.n_topics, p=topic_probs)
                
                self.topic_assignments[doc_id][word_idx] = new_topic
                self.n_wd[doc_id, word_id] += 1
                self.n_td[doc_id, new_topic] += 1
                self.n_t[new_topic] += 1
    
    def get_topic_word_distribution(self):
        word_distribution = np.zeros((self.n_topics, self.n_words))
        for doc_id in range(self.n_docs):
            for word_idx, word in enumerate(self.documents[doc_id]):
                topic = self.topic_assignments[doc_id][word_idx]
                word_id = self.word_to_id[word]
                word_distribution[topic, word_id] += 1
        return word_distribution / word_distribution.sum(axis=1)[:, np.newaxis]
    
    def get_document_topic_distribution(self):
        return (self.n_td + self.alpha) / (self.n_td.sum(axis=1)[:, np.newaxis] + self.n_topics * self.alpha)

def load_conversation_data(file_path):
    """Load conversation data from a .txt file where each utterance is followed by <EOS>"""
    with open(file_path, 'r',  encoding="utf-8") as file:
        content = file.read()
    
    # Split by <EOS> and clean up
    utterances = content.split("<EOS>")
    utterances = [re.sub(r'\s+', ' ', utt.strip()) for utt in utterances if utt.strip()]
    
    # Tokenize each utterance into words
    tokenized_utterances = [utt.split() for utt in utterances]
    
    return tokenized_utterances