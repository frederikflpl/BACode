import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1

import AspectFinder
import GraphInference


def word_feats(words):
    return dict([(word, True) for word in words])

aspects = AspectFinder.find_aspects()

minus = [f for f in aspects if f[1][0]=="-"]
plus = [f for f in aspects if f[1][0]=="+"]

sentences = [' '.join(s) for s in product_reviews_1.sents()]

minusfeats = [(word_feats(s.split()), '-') for s in sentences for f in minus if s.find(f[0])!=-1]
plusfeats = [(word_feats(s.split()), '+') for s in sentences for f in plus if s.find(f[0])!=-1]

minusfeats1 = [(s.split(), '-') for s in sentences for f in minus if s.find(f[0])!=-1]
plusfeats1 = [(s.split(), '+') for s in sentences for f in plus if s.find(f[0])!=-1]

minuscutoff = int(len(minusfeats)*3/4)
pluscutoff = int(len(plusfeats)*3/4)

trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
trainfeats1 = minusfeats1[:minuscutoff] + plusfeats1[:pluscutoff]
testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

trainfeats1 = [(i, i>49) for i in range(100)]
unlabeledfeats = []
trainfeats = GraphInference.inferGraph(trainfeats1, unlabeledfeats)


classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()