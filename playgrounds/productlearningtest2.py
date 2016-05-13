import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1

def word_feats(words):
    return dict([(word, True) for word in words])

negids = product_reviews_1.fileids('-1')
posids = product_reviews_1.fileids('+1')

negfeats = [(word_feats(product_reviews_1.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(product_reviews_1.words(fileids=[f])), 'pos') for f in posids]

negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()