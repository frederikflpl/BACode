import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1

def word_feats(words):
    return dict([(word, True) for word in words])

negfeatures = [(word_feats(f[0]),'neg') for f in product_reviews_1.features() if f[1][0]=="-"]
posfeatures = [(word_feats(f[0]),'pos') for f in product_reviews_1.features() if f[1][0]=="+"]

print(len(negfeatures))
print(len(posfeatures))

negcutoff = int(len(negfeatures)*3/4)
poscutoff = int(len(posfeatures)*3/4)

trainfeats = negfeatures[:negcutoff] + posfeatures[:poscutoff]
testfeats = negfeatures[negcutoff:] + posfeatures[poscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
#classifier.show_most_informative_features()
print(str(classifier.classify(word_feats(product_reviews_1.reviews()[0].features()) )))
word_list = []
for review in product_reviews_1.reviews():
    for sent in review.sents():
        print(sent)
        for word in sent:
            word_list.append(word)
tagged_word_list = nltk.pos_tag(word_list)
nouns = [w[0] for w in tagged_word_list if w[1]=='NN']
print(nltk.FreqDist(nouns).most_common(10))

# negids = movie_reviews.fileids('neg')
# posids = movie_reviews.fileids('pos')
#
# negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
# posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
#
# negcutoff = int(len(negfeats)*3/4)
# poscutoff = int(len(posfeats)*3/4)
#
# trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
# testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
# print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
#
# classifier = NaiveBayesClassifier.train(trainfeats)
# print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
# classifier.show_most_informative_features()