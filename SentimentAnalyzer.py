import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1
from nltk import precision, recall
import AspectFinder
import collections


def word_feats(words):
    return dict([(word, True) for word in words])

def evaluate_classifier(classifier):
    aspects = AspectFinder.AspectFinder().get_aspects()

    minus = [f for f in aspects if f[1][0]=="-"]
    plus = [f for f in aspects if f[1][0]=="+"]

    sentences = [' '.join(s) for s in product_reviews_1.sents()]

    minusfeats = [(word_feats(s.split()), '-') for s in sentences for f in minus if s.find(f[0])!=-1]
    plusfeats = [(word_feats(s.split()), '+') for s in sentences for f in plus if s.find(f[0])!=-1]

    minuscutoff = int(len(minusfeats)*3/4)
    pluscutoff = int(len(plusfeats)*3/4)

    trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
    testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))


    classifier = train(classifier, trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', precision(refsets['pos'], testsets['pos']))
    print('pos recall:', recall(refsets['pos'], testsets['pos']))
    print('neg precision:', precision(refsets['neg'], testsets['neg']))
    print('neg recall:', recall(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features()
    print(classifier.classify(word_feats(["I", "hate", "it", "."])))
    print(classifier.classify(word_feats(["I", "love", "it", "."])))

def train(classifier, trainfeats, traintargets=[]):
    try:
        classifier = classifier.train(trainfeats)
    except AttributeError:
        classifier = classifier.fit(trainfeats, traintargets)

    return classifier

if __name__ == '__main__':
    evaluate_classifier(NaiveBayesClassifier)