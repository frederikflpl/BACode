import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1

def word_feats(words):
    return dict([(word, True) for word in words])

minus1 = [f for f in product_reviews_1.features() if f[1]=="-1"]
minus2 = [f for f in product_reviews_1.features() if f[1]=="-2"]
minus3 = [f for f in product_reviews_1.features() if f[1]=="-3"]
plus1 = [f for f in product_reviews_1.features() if f[1]=="+1"]
plus2 = [f for f in product_reviews_1.features() if f[1]=="+2"]
plus3 = [f for f in product_reviews_1.features() if f[1]=="+3"]


sentences = [' '.join(s) for s in product_reviews_1.sents()]

minus1feats = [(word_feats(s.split()), '-1') for s in sentences for f in minus1 if s.find(f[0])!=-1]
minus2feats = [(word_feats(s.split()), '-2') for s in sentences for f in minus2 if s.find(f[0])!=-1]
minus3feats = [(word_feats(s.split()), '-3') for s in sentences for f in minus3 if s.find(f[0])!=-1]
plus1feats = [(word_feats(s.split()), '+1') for s in sentences for f in plus1 if s.find(f[0])!=-1]
plus2feats = [(word_feats(s.split()), '+2') for s in sentences for f in plus2 if s.find(f[0])!=-1]
plus3feats = [(word_feats(s.split()), '+3') for s in sentences for f in plus3 if s.find(f[0])!=-1]

minus1cutoff = int(len(minus1feats)*3/4)
minus2cutoff = int(len(minus2feats)*3/4)
minus3cutoff = int(len(minus3feats)*3/4)
plus1cutoff = int(len(plus1feats)*3/4)
plus2cutoff = int(len(plus2feats)*3/4)
plus3cutoff = int(len(plus3feats)*3/4)

trainfeats = minus1feats[:minus1cutoff] + plus1feats[:plus1cutoff] + \
             minus2feats[:minus2cutoff] + plus2feats[:plus2cutoff] + \
             minus3feats[:minus3cutoff] + plus3feats[:plus3cutoff]
testfeats = minus1feats[minus1cutoff:] + plus1feats[plus1cutoff:] + \
             minus2feats[minus2cutoff:] + plus2feats[plus2cutoff:] + \
             minus3feats[minus3cutoff:] + plus3feats[plus3cutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()