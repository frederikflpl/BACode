from nltk import FreqDist, ConditionalFreqDist, BigramAssocMeasures
from nltk.corpus import product_reviews_1, product_reviews_2


def prepareReviews(neutral=False):
    reviews = product_reviews_1.reviews() + product_reviews_2.reviews()
    reviewlines = []
    for review in reviews:
        for line in review.review_lines:
            reviewlines.append(line)

    # minusfeatsann = [(s, '-') for s in sentences for f in minus if s.find(f[0]) != -1]
    # plusfeatsann = [(s, '+') for s in sentences for f in plus if s.find(f[0]) != -1]
    best = bestWords()
    sentfeats = [[(' '.join(bestfeats(line.sent,best)), f[1][0]) for f in line.features] for line in reviewlines if
                 len(line.features) > 0]
    plusfeatsann = []
    minusfeatsann = []
    for sentfeat in sentfeats:
        for feat in sentfeat:
            if feat[1] == "+":
                plusfeatsann.append(feat)
            elif feat[1] == "-":
                minusfeatsann.append(feat)

    if neutral:
        neutralfeats = [' '.join(bestfeats(line.sent,best)) for line in reviewlines if len(line.features) == 0]

    if len(minusfeatsann)>len(plusfeatsann):
        minusfeatsann = minusfeatsann[:len(plusfeatsann)]
    else:
        plusfeatsann = plusfeatsann[:len(minusfeatsann)]

    if neutral:
        minlen = min(len(minusfeatsann),len(plusfeatsann),len(neutralfeats))
        minusfeatsann = minusfeatsann[:minlen]
        plusfeatsann = plusfeatsann[:minlen]
        neutralfeats = neutralfeats[:minlen]

    minusfeats = [f[0] for f in minusfeatsann]
    plusfeats = [f[0] for f in plusfeatsann]

    minustargets = [f[1] for f in minusfeatsann]
    plustargets = [f[1] for f in plusfeatsann]

    if neutral:
        neutraltargets = ["~" for f in neutralfeats]
        neutralcutoff = int(len(neutralfeats) * 3 / 4)

    minuscutoff = int(len(minusfeats) * 3 / 4)
    pluscutoff = int(len(plusfeats) * 3 / 4)

    trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
    traintargets = minustargets[:minuscutoff] + plustargets[:pluscutoff]
    testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
    testtargets = minustargets[minuscutoff:] + plustargets[pluscutoff:]

    if neutral:
        trainfeats += neutralfeats[:neutralcutoff]
        traintargets += neutraltargets[:neutralcutoff]
        testfeats += neutralfeats[neutralcutoff:]
        testtargets += neutraltargets[neutralcutoff:]

    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
    return [trainfeats, traintargets, testfeats, testtargets]


def bestWords(neutral=False):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    reviews = product_reviews_1.reviews() + product_reviews_2.reviews()
    reviewlines = []
    for review in reviews:
        for line in review.review_lines:
            reviewlines.append(line)

    featlines = [line for line in reviewlines if len(line.features) > 0]
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

    if neutral:
        neutrallines = [line for line in reviewlines if len(line.features) == 0]
        neutralwords = []
        for line in neutrallines:
            for word in line.sent:
                neutralwords.append(word)
        for word in neutralwords:
            word_fd[word.lower()] += 1
            label_word_fd['~'][word.lower()] += 1
        neu_word_count = label_word_fd['~'].N()

    for word in pluswords:
        word_fd[word.lower()] += 1
        label_word_fd['+'][word.lower()] += 1

    for word in minuswords:
        word_fd[word.lower()] += 1
        label_word_fd['-'][word.lower()] += 1

    pos_word_count = label_word_fd['+'].N()
    neg_word_count = label_word_fd['-'].N()
    total_word_count = pos_word_count + neg_word_count

    if neutral:
        total_word_count += neu_word_count

    word_scores = {}

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['+'][word],
                                               (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['-'][word],
                                               (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

        if neutral:
            neu_score = BigramAssocMeasures.chi_sq(label_word_fd['~'][word],
                                               (freq, neu_word_count), total_word_count)
            word_scores[word] += neu_score

    #best = sorted(word_scores.items(), key=(lambda s: s[1]), reverse=True)[:515]
    best = sorted(word_scores.items(), key=(lambda s: s[1]), reverse=True)[:1000]
    #best = sorted(word_scores.items(), key=(lambda s: s[1]), reverse=True)[:200]
    #best = sorted(word_scores.items(), key=(lambda s: s[1]), reverse=True)[:2000]
    return set([w for w, s in best])

def bestfeats(feats,best):
    bestf = [f for f in feats if f in best]
    return bestf