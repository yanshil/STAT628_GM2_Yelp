
"""
doc2vec_model.py: Train Model with doc2vec
"""

__author__ = "Yanshi Luo", "Peijin Li"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import pandas as pd
import process_text

train_filename = 'train_data.csv'
trainDF = pd.read_csv(train_filename)

isUnitTest = True

if isUnitTest:
    trainDF = trainDF.head(500)

n_train = trainDF.shape[0]

trainDF['tokens'] = [review.split() for review in process_text.process_reviews(trainDF.text)]

"""
Training Doc2vec
"""
documents = []
for i in range(0, n_train):
    documents.append(LabeledSentence(words=trainDF.tokens[i], tags=[trainDF.stars[i]]))

model = Doc2Vec(vector_size=50, window=10, min_count=10, workers=4)
# vector_size (int) – Dimensionality of the feature vectors.
# window (int) – The maximum distance between the current and predicted word within a sentence.
# min_count (int) – Ignores all words with total frequency lower than this.
# workers (int) – Use these many worker threads to train the model (=faster training with multicore machines).
model.build_vocab(documents)
model.train(documents, total_examples=n_train, epochs=10)
print(len(model.wv.vocab))
model.save('./yelp.doc2vec')

# model.infer_vector(trainDF.tokens[0])
# model.docvecs[trainDF.stars[0]]
# model.wv.vocab


