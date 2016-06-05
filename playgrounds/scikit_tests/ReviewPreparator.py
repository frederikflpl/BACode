from nltk.corpus import product_reviews_1
import AspectFinder

trainfeats = []
traintargets = []
testfeats = []
testtargets = []

def prepareReviews():
    reviews = product_reviews_1.reviews()
    reviewlines = []
    for review in reviews:
        for line in review.review_lines:
            reviewlines.append(line)

    # minusfeatsann = [(s, '-') for s in sentences for f in minus if s.find(f[0]) != -1]
    # plusfeatsann = [(s, '+') for s in sentences for f in plus if s.find(f[0]) != -1]
    sentfeats = [[(' '.join(line.sent), f[1][0]) for f in line.features] for line in reviewlines if
                 len(line.features) > 0]
    plusfeatsann = []
    minusfeatsann = []
    for sentfeat in sentfeats:
        for feat in sentfeat:
            if feat[1] == "+":
                plusfeatsann.append(feat)
            elif feat[1] == "-":
                minusfeatsann.append(feat)

    if len(minusfeatsann)>len(plusfeatsann):
        minusfeatsann = minusfeatsann[:len(plusfeatsann)]
    else:
        plusfeatsann = plusfeatsann[:len(minusfeatsann)]

    minusfeats = [f[0] for f in minusfeatsann]
    plusfeats = [f[0] for f in plusfeatsann]

    minustargets = [f[1] for f in minusfeatsann]
    plustargets = [f[1] for f in plusfeatsann]

    minuscutoff = int(len(minusfeats) * 3 / 4)
    pluscutoff = int(len(plusfeats) * 3 / 4)

    trainfeats = minusfeats[:minuscutoff] + plusfeats[:pluscutoff]
    traintargets = minustargets[:minuscutoff] + plustargets[:pluscutoff]
    testfeats = minusfeats[minuscutoff:] + plusfeats[pluscutoff:]
    testtargets = minustargets[minuscutoff:] + plustargets[pluscutoff:]
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
    return [trainfeats, traintargets, testfeats, testtargets]
def prepareReviewData():
    return trainfeats

def prepareReviewTargets():
    return traintargets

def prepareReviewTestData():
    return ['Hello']

def prepareReviewTestTargets():
    return ['Greeting']