import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1
from nltk import precision, recall
import collections

def word_feats(words):
    return dict([(word, True) for word in words])

reviews = product_reviews_1.reviews()
reviewlines = []
for review in reviews:
    for line in review.review_lines:
        reviewlines.append(line)


sentfeats = [[(word_feats(line.sent), f[1][0]) for f in line.features] for line in reviewlines if len(line.features)>0]
plusfeats = []
minusfeats = []
for sentfeat in sentfeats:
    for feat in sentfeat:
        if feat[1] == "+":
            plusfeats.append(feat)
        elif feat[1] == "-":
            minusfeats.append(feat)



if len(minusfeats) > len(plusfeats):
    minusfeats = minusfeats[:len(plusfeats)]
else:
    plusfeats = plusfeats[:len(minusfeats)]

minuscutoff = int(len(minusfeats)*3/4)
pluscutoff = int(len(plusfeats)*3/4)

trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
print('pos precision:', precision(refsets['+'], testsets['+']))
print('pos recall:', recall(refsets['+'], testsets['+']))
print('neg precision:', precision(refsets['-'], testsets['-']))
print('neg recall:', recall(refsets['-'], testsets['-']))
classifier.show_most_informative_features()
print(classifier.classify(word_feats(["I", "hate", "it", "."])))
print(classifier.classify(word_feats(["I", "love", "it", "."])))