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


sentfeats = [[(word_feats(line.sent), f[1]) for f in line.features] for line in reviewlines if len(line.features)>0]
plus1feats = []
plus2feats = []
plus3feats = []
minus1feats = []
minus2feats = []
minus3feats = []
for sentfeat in sentfeats:
    for feat in sentfeat:
        if feat[1] == "+1":
            plus1feats.append(feat)
        elif feat[1] == "+2":
            plus2feats.append(feat)
        elif feat[1] == "+3":
            plus3feats.append(feat)
        elif feat[1] == "-1":
            minus1feats.append(feat)
        elif feat[1] == "-2":
            minus2feats.append(feat)
        elif feat[1] == "-3":
            minus3feats.append(feat)
# old accuracy: unbalanced: 0.3569208954323297 actual: 0.17901234567901234
# new accuracy: unbalanced: 0.24524714828897337 actual: 0.2807017543859649

feats = [minus1feats,minus2feats,minus3feats,plus1feats,plus2feats,plus3feats]
featlens = [len(featlist) for featlist in feats]
minfeatlen = min(featlens)
print(minfeatlen," ",max(featlens))

minus1feats = minus1feats[:minfeatlen]
minus2feats = minus2feats[:minfeatlen]
minus3feats = minus3feats[:minfeatlen]
plus1feats = plus1feats[:minfeatlen]
plus2feats = plus2feats[:minfeatlen]
plus3feats = plus3feats[:minfeatlen]

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
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
print('+1 precision:', precision(refsets['+1'], testsets['+1']))
print('+1 recall:', recall(refsets['+1'], testsets['+1']))
print('+2 precision:', precision(refsets['+2'], testsets['+2']))
print('+2 recall:', recall(refsets['+2'], testsets['+2']))
print('+3 precision:', precision(refsets['+3'], testsets['+3']))
print('+3 recall:', recall(refsets['+3'], testsets['+3']))
print('-1 precision:', precision(refsets['-1'], testsets['-1']))
print('-1 recall:', recall(refsets['-1'], testsets['-1']))
print('-2 precision:', precision(refsets['-2'], testsets['-2']))
print('-2 recall:', recall(refsets['-2'], testsets['-2']))
print('-3 precision:', precision(refsets['-3'], testsets['-3']))
print('-3 recall:', recall(refsets['-3'], testsets['-3']))
classifier.show_most_informative_features()
print(classifier.classify(word_feats(["I", "hate", "it", "."])))
print(classifier.classify(word_feats(["I", "love", "it", "."])))