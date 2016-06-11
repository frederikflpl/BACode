import itertools
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1, stopwords
from nltk import precision, recall, BigramAssocMeasures, BigramCollocationFinder, FreqDist, ConditionalFreqDist
import collections



def evaluate_classifier(featx):
    reviews = product_reviews_1.reviews()
    reviewlines = []
    for review in reviews:
        for line in review.review_lines:
            reviewlines.append(line)


    sentfeats = [[(featx(line.sent), f[1][0]) for f in line.features] for line in reviewlines if len(line.features)>0]
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

    #return nltk.classify.util.accuracy(classifier, testfeats)
    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', precision(refsets['+'], testsets['+']))
    print('pos recall:', recall(refsets['+'], testsets['+']))
    print('neg precision:', precision(refsets['-'], testsets['-']))
    print('neg recall:', recall(refsets['-'], testsets['-']))
    classifier.show_most_informative_features()
    print(classifier.classify(featx(["I", "hate", "it", "."])))
    print(classifier.classify(featx(["I", "love", "it", "."])))


def word_feats(words):
    return dict([(word, True) for word in words])


def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])


def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d


stopset = set(stopwords.words('english'))

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()


reviews = product_reviews_1.reviews()
reviewlines = []
for review in reviews:
    for line in review.review_lines:
        reviewlines.append(line)


featlines = [line for line in reviewlines if len(line.features)>0]
pluswords = []
minuswords = []
for line in featlines:
    plus = False
    minus = False
    for feat in line.features:
        if feat[1][0] == "+":
            plus = True
        elif feat[1][0] == "-":
            minus = True
    if plus:
        for word in line.sent:
            pluswords.append(word)
    if minus:
        for word in line.sent:
            minuswords.append(word)


for word in pluswords:
    word_fd[word.lower()]+=1
    label_word_fd['+'][word.lower()]+=1

for word in minuswords:
    word_fd[word.lower()]+=1
    label_word_fd['-'][word.lower()]+=1

pos_word_count = label_word_fd['+'].N()
neg_word_count = label_word_fd['-'].N()
total_word_count = pos_word_count + neg_word_count

word_scores = {}

for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['+'][word],
                                           (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['-'][word],
                                           (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

# accs = []
# for x in range(0,2000,5):
#print(x)
best = sorted(word_scores.items(), key=(lambda s: s[1]), reverse=True)[:515]
bestwords = set([w for w, s in best])
# 1500: accuracy: 0.8091397849462365
# pos precision: 0.891156462585034
# pos recall: 0.7043010752688172
# neg precision: 0.7555555555555555
# neg recall: 0.9139784946236559
# 2000: accuracy: 0.803763440860215
# pos precision: 0.864516129032258
# pos recall: 0.7204301075268817
# neg precision: 0.7603686635944701
# neg recall: 0.8870967741935484
# 1000: accuracy: 0.8064516129032258
# pos precision: 0.8851351351351351
# pos recall: 0.7043010752688172
# neg precision: 0.7544642857142857
# neg recall: 0.9086021505376344
# 500: accuracy: 0.8252688172043011
# pos precision: 0.8622754491017964
# pos recall: 0.7741935483870968
# neg precision: 0.7951219512195122
# neg recall: 0.8763440860215054
# 515: accuracy: 0.8306451612903226
# pos precision: 0.863905325443787
# pos recall: 0.7849462365591398
# neg precision: 0.8029556650246306
# neg recall: 0.8763440860215054


print('evaluating best word features')
evaluate_classifier(best_word_feats)
#accs.append((x,evaluate_classifier(best_word_feats)))

#print("max",max(accs,key=(lambda s: s[1])),"min",min(accs,key=(lambda s: s[1])))

#print('evaluating best words + bigram chi_sq word features')
#evaluate_classifier(best_bigram_word_feats)


