import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1

def word_feats(words):
    return dict([(word, True) for word in words])

minus = [f for f in product_reviews_1.features() if f[1][0]=="-"]
plus = [f for f in product_reviews_1.features() if f[1][0]=="+"]


sentences = [' '.join(s) for s in product_reviews_1.sents()]

minusfeats = [(word_feats(s.split()), '-') for s in sentences for f in minus if s.find(f[0])!=-1]
plusfeats = [(word_feats(s.split()), '+') for s in sentences for f in plus if s.find(f[0])!=-1]

minuscutoff = int(len(minusfeats)*3/4)
pluscutoff = int(len(plusfeats)*3/4)

trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()