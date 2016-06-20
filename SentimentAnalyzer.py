import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import product_reviews_1
from nltk import precision, recall
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
import AspectFinder
import collections
from playgrounds.scikit_tests import classifiertest
from nltk import sent_tokenize


def prepare_classifier(classifier):
    classifiertest.getData()
    classifiertest.fit(classifier)


def classify(reviews, aspects, classifier):
    try:
        reviewsents = sent_tokenize(reviews)
    except Exception:
        reviews = ' '.join(reviews)
        reviewsents = sent_tokenize(reviews)
    feats = [(a, f) for f in reviewsents for a in aspects if f.find(a)!=-1]
    ascores = {a: [0,0] for a in aspects}
    for feat in feats:
        print(feat[0])
        print(feat[1])
        predicted = classifier.predict([feat[1]])
        print(predicted, "\n")
        ascores[feat[0]][1] += 1
        if (predicted=="+"):
            ascores[feat[0]][0]+=1
        elif (predicted=="-"):
            ascores[feat[0]][0] -= 1
    aggscores = [(a,ascores[a][0]/ascores[a][1]) for a in aspects]
    print(aggscores)


if __name__ == '__main__':
    classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), binary=True)),
                          ('clf', BernoulliNB(alpha=1, fit_prior=True)),
                          ])
    prepare_classifier(classifier)
    reviews = """An engineer's degree is an advanced academic degree in engineering that is conferred in Europe, some countries of Latin America, and a few institutions in the United States.

In Europe, the engineer degree is ranked at the same academic level as a bachelor's degree or master's degree, and is often known literally as an "engineer diploma" (abbreviated Dipl.-Ing. or DI).
In some countries of Latin America and the United States, the engineer's degree can be studied after the completion of a master's degree and is usually considered higher than the master's degree
but below the doctorate in engineering (abbreviated Dr.-Ing.) in Europe. In other countries of Latin America, there is no proper engineer's degree,
but the title of Engineer is used for 5 year bachelor's graduates."""
    aspects = ["degree", "engineer", "bachelor", "Latin America"]
    classify(reviews, aspects, classifier)
    reviews = """Amazon released the first Kindle in November of 2007. Eight and a half years later, the Oasis model is the 8th generation Kindle and the third evolution of Kindle with built-in illumination. The price of $300+ was a bit of a shock, but I ordered mine as soon as it was announced on April 13th. I then spent the next two weeks reading what was available online describing the features and benefits and asking myself whether or not I had made the right decision. Now that it’s finally been released and I have it in my hands, how different is it, and is it worth what appears at first to be a very high price?

(My Kindle experience: I purchased my first Kindle (a first generation Paperwhite), when it was first released in Sept 2012. I purchased the Voyage when it was first released, in Sept 2014. I use my Kindles every day. I’ll compare this new Oasis to both of those models.)

I've decided to give the Oasis 5 stars based upon the design and size which I find exceptionally nice. I have no issues with the display on my Oasis although I've read a few early reports that do indicate some units may have display problems. The price may be reason for some people to consider this a less than 5-star product. For me personally I use my Kindle every day, it's a constant companion, and I have found no faults with this model and enough improvements from the Voyage to make it preferable.

The price of the Oasis is high enough that this is not a casual purchase for most people, myself included. As I write this review, the regular (non-illuminated) Kindle can be purchased for as little as $59.99! That is remarkable. The Voyage and the latest Paperwhite are both exceptionally good e-readers, provide essentially the same display and almost the same reading experience, and both are considerably less expensive. Amazon has developed a product line range of four models at different price points so that one or another are probably affordable for just about anyone. They also have tried to provide enough differentiation in features from one model to the next to make the price differences worthwhile. It's like buying a new car - do you want to get the most affordable car for basic transportation, or are you willing to spent more money to get a high-end sound system, leather upholstery, and navigation?

I'll go through each aspect of the Oasis and try to explain what is different about it, and what is not. In the end, some people will consider the Oasis as money well spent, and others will be happy with what they already have. And that is how it should be!

My thoughts in much more detail follow.

AMAZON's DESIGN OBJECTIVES FOR THE KINDLE OASIS

In reading about the Oasis since it was announced, one thing became clear: Amazon wants this new Kindle to 'disappear' into your hands, so that you forget about the Kindle and get lost in the book you are reading.

Their objectives were therefore light weight and thinness above all, along with improving their already premium display if possible and providing the best possible battery life. I think that it's important to keep those design objectives in mind when evaluating the Oasis.

My own impressions were probably influenced by knowing these things ahead of time, but I must say that the Oasis is very significantly lighter and thinner than any of my other Kindles, and I like the style with the wider side for gripping. The display is superb, following the already excellent Voyage, with an improved illumination design.

INITIAL (NON-CUSTOMER) REACTIONS TO THE OASIS (MOSTLY CRITICAL?)

Early articles and reviews seem to have been more negative than positive. Most have mentioned one or more of the following points as concerns:
-- High price
-- Same 6” display size as other Kindles (some people want larger display)
-- No audio capability (none of the current e-ink Kindles have audio, see note below as well as comments to this review for elaboration)
-- No Bluetooth
-- No color display (e-ink technology as used in all Kindles is not yet developed with color, as far as I know)
-- Not waterproof

This new Kindle does not offer any of the above, and yet it is now the most expensive Kindle model. What does it offer?

ADDED NOTE 5/28/16: Amazon is now offering an audio adaptor for some Kindle models that enables VoiceView text-to-speech capability. This is not the same as listening to Audible books. See the following links:

- Kindle Paperwhite Blind and Visually Impaired Readers Bundle (Paperwhite plus audio adaptor)
- Kindle Audio Adapter (audio adaptor alone)
- Amazon's announcement can be read here:[...]
- Fire help page explaining VoiceView capability: https://www.amazon.com/gp/help/customer/display.html?nodeId=201829340

USING THE OASIS

The Oasis is shaped differently from other recent Kindles, has the weight shifted to one side, but retains the same display as the Voyage and Paperwhite with some improvements made to the lighting. How is it to use?

-- The shape and size is different, and I’m quickly finding that it feels more natural to hold. The wider side is intended to be where you grip it, and if you hold it with left or right hand it re-orients the display automatically. The grip is wider and fatter than the other Kindles. Amazon refers to it as an ‘ergonomic’ grip.
-- Having a wider side with the page turn buttons is considerably more user friendly as well. I always found the Voyage page turn buttons to be difficult to avoid pressing by accident since the sides of the Kindle were very narrow and it was hard to hold the Voyage without touching the page turn buttons. Not so with the Oasis, the buttons are more prominent and easier to either find, or to avoid, and there is plenty of room to hold the Kindle without touching them by accident
-- Holding the Oasis with the cover is comfortable, but what is really impressive is how light it is without the cover. And it’s very easy to detach the Oasis from the cover, much easier than the Paperwhite in particular (the Voyage also slips out of its cover quite easily).
-- If you like the way that the Voyage ‘origami’ cover can be used to stand up the Kindle for reading, then that’s obviously not a feature of this new design. But I know that many people prefer a book style cover anyway, like I have with my Paperwhite, and this cover returns to that style.
-- Display can be set to landscape or portrait via the settings menu.
-- Oasis does NOT have the adaptive light sensor that the Voyage incorporates. You adjust the brightness of the display manually, a simple and quick adjustment.
-- The power button easier to use. With the Voyage if you have the origami cover and fold the cover back to read, then it covers up the power button which is on the rear of the device.
-- The page turn buttons are raised and easier to sense with your thumb than the buttons on the Voyage. There is a very slight click when the buttons are pushed.

My thoughts:

-- I miss the 'origami' cover of my Voyage. Some people prefer the book-style cover, but I like how the origami cover allows the Voyage to be propped up for reading. That's not possible with the Oasis, at least not with the current cover.
-- I do like the feel of the Oasis in my hands. The wide side used for gripping is a big improvement - see the video. I'll need to use it for a longer time in order to see if it really makes a big difference for reading, but my initial impression is that it's much nicer.

THE OASIS ‘SYSTEM’

The Kindle Oasis is not simply an e-ink reader, it is a reader plus cover and with the two designed to work together. The Oasis without cover is light and thin, shaped differently from other Kindles, with a display that automatically ‘rotates’ so that holding it in either the left or right hand will still result in an upright display. Both the Oasis and the cover incorporate batteries, and the two work together to give the ability to use the Oasis for long periods of time between charges.

The cover attaches and detaches easily, and is held in place with magnets and very secure. It really is convenient to remove the cover when you want to hold the Oasis for reading and enjoy the light weight and thin size, and when the cover is replaced then the battery in the Oasis automatically begins recharging from the larger battery within the cover. It is a very clever system and it works well, and transparently to the user.

THE OASIS DISPLAY

Uniformness of the lighting was a chronic complaint for the early self illuminated Kindles, particularly the Paperwhite when it was first introduced. The Voyage screen and illumination was a step forward from the Paperwhite, and the Paperwhite itself is now in it's third generation. At this point, the display specifications for the Oasis are the same as the Voyage and Paperwhite, but Amazon says that the lighting design is improved. It has what I believe are 10 LED's along the wider side of the display, but they are very very difficult to discern even when looking at a sharp angle. I am able to see some shadowing from the LED's under certain conditions, but it is very subtle. Really, the display in my Oasis is faultless - crisp, sharp and bright. It is probably even better than the Voyage although my Voyage display is also quite excellent. My Paperwhite does have a very noticeable shadowing from the illumination which in the case of that model is coming from the bottom of the display. In the three and a half years since the first Paperwhite was released Amazon has really improved the display to the point where it is truly excellent in all respects.

One change is that the Oasis display does not include the 'adaptive' light feature of the Voyage. That adjusts the light setting depending upon the ambient light, and in my experience with my Voyage it is sometimes a good feature but not always fully adjusting how I prefer. Amazon decided to eliminate that feature for this new model, and I doubt that I'll miss it because manually adjusting the screen brightness is a very easy thing to do, and I was always messing with the adjustment on my Voyage anyway because the automatic adjustment often was not what I preferred.

Overall though, in my initial use I find little difference between the Oasis and my other Kindles (with respect to the display), but that is not a negative. My Voyage screen has been without fault since I first received it. I find the display to be clear and sharp and the range of illumination is very wide, sufficient for reading in the dark and also to illuminate the screen very adequately in bright light.

SIZE AND WEIGHT

The Oasis is shaped differently than other Kindles (a bit wider, and shorter) although the screen size is the same (6”). About two-thirds of the width of the Oasis is incredibly thin – less than half the thickness of the Voyage and almost a third the thickness of the Paperwhite. The weight of the Oasis by itself is quite a bit lighter than either of the other models, and even with the battery cover attached it is significantly less than the other models with cover:
-- Oasis: 4.6 oz without cover + 3.8 oz for cover = 8.4 oz total
-- Voyage: 6.3 oz without cover + 4.8 oz (origami leather cover) = 11.1 oz total
-- Paperwhite: 7.2 oz without cover + 4.7 oz (Amazon leather cover) = 11.9 oz total
-- Kindle: 6.7 oz without cover + 3.8 oz (Amazon leather cover) = 10.5 oz total

My thoughts:
-- The display is really impressively thin – noticeably less than the Voyage. It's remarkably thin when holding it.
-- What I find particularly impressive is how light and easy to hold this Oasis is by itself. The design places the weight closer to your grip (20% closer according to Amazon) and it does feel more comfortable and ‘like a book’.
-- Amazon’s goal – for the Oasis to ‘disappear’ in your hand – is not something I can quite confirm yet. It’s a bit of hyperbole, really, but the intent is there, and this Oasis is actually so light to hold that I can see this as not so much of an exaggeration, once you have used it for a while and are simply relaxing and reading a book with it.

PRICING

My first reaction to the price was that it sure sounded like a lot - $289.99 for the least expensive model. Later I took the time to compare it on an apples-to-apples basis to the other Kindle models, and here is what I found:
-- For comparison the pricing here is for Wi-Fi only, with special offers, and including Amazon’s own leather cover for the respective models (Wi-Fi plus 3G is +$70 for all Kindles, add $20 to get without special offers)
-- $290 – Oasis, price includes leather cover
-- $260 – Voyage plus Amazon’s leather cover ($200 + 60 = $260, or $30 less) (unchanged since first announced)
-- $160 – Paperwhite plus Amazon’s leather cover ($120 + 40 = $160, or $130 less) (also unchanged)
-- $120 – Kindle plus Amazon’s leather cover ($80 + 40 = $120, or $170 less)

Notwithstanding the above, it’s clear that the Oasis itself is as much as $210 more than the lease expensive Kindle. You can buy three base model Kindles (without illuminated screen) for the price of one Oasis, even including the cost of cheap covers for each of them.

My thoughts:
-- The cost is high but depending upon how you would expect to purchase your Kindle, it may not be quite as bad as it first appears.
-- The main difference is, with the Oasis you that don’t have a choice, you MUST purchase it with the leather cover because the Oasis and cover are designed to work together as a ‘system’ (see above). With the Voyage and Paperwhite you can purchase the Kindle without the cover, and you can also purchase much less expensive non-Amazon and non-leather covers.
-- If you would normally buy a nice ($40-60) cover for your Kindle, then the Oasis may not be that much more than that you'd pay for a Voyage. If you don't use a cover, or you would normally buy a less expensive non-leather non-Amazon cover, then the price is much higher than you'd pay for one of the other models.
-- If value for your money is first consideration, the either the Paperwhite (if you want illuminated screen), or the base Kindle, is clearly the best choice, at either $120 or $80 plus the price of the cover of your choice.

These prices are all normal full retail prices. Amazon has been discounting the other Kindle models recently, so the differences have been even greater.

Oasis buyers are probably looking for the most premium e-reader, want the latest and greatest, and are comfortable paying for it. This is not the Kindle model intended for budget purchasers.

BATTERY CAPACITY AND LIFE

Amazon does not give the actual battery capacity (in mAh) in their specs although eventually that information should be available online and I'll add it to this review. In the meantime Amazon does state how long the various Kindle models will operate on battery, and a comparison can be made. This is something I wanted to do for myself since the battery arrangement of the Oasis 'system' is so different for other Kindles, and because some of the early reports suggested very long battery life.

Here is what battery life actually works out to be, in terms of actual available reading time between charges for the current Kindle models, using Amazon's own stated specs which all assume "wireless off and the light setting at 10":

-- Kindle: 4 weeks @ ½ hr of reading/day = 14 hrs
-- Paperwhite or Voyage: 6 weeks @ ½ hr of reading/day = 21 hrs
-- Oasis (including cover): 8 or 9 weeks @ ½ hr of reading/day = 28 - 31.5 hrs
-- Oasis (without cover): 2 weeks @ ½ hr of reading/day = 7 hrs

The differences are significant and the Oasis has a longer possible use between charge to be sure (making use of the battery in the cover). Whether or not that is a really important difference, I think it will depend on how much someone wants to use their Kindle between needing to plug it in to recharge fully.

NEW INFORMATION added 5/1/16, 5/22/16: One website now has a teardown report and states that the Oasis battery is 245 mAh size. A further report (update 5/22) gives the battery size in the cover as 1290 mAh, for a total of 1535 mAh. Compare that to the 1320 mAh battery included in the Voyage or the 1420 mAh battery in the Paperwhite and it's clear that the Oasis really must rely upon the cover in order to get respectable battery life, but if these reports are correct then the total battery capacity in the Oasis is larger than either of those models. The larger battery plus improved battery management software would explain Amazon's claim of longer available reading time as I've summarized above. For those interested in more thoughts on this battery arrangement, please see comments to this review, below.

MORE NEW INFORMATION added 5/14/16: I've added a photo to this review that shows how the Oasis will go into 'hibernation' mode after sleeping for some period of time. When waking up from hibernation, the display shows 'waking up' at the bottom and takes a couple of seconds longer before it fully wakes up. This is a new feature of the Oasis and I'm sure it is part of the battery management software it incorporates, to deal with the different battery arrangement and give the best life between charges.

Other details regarding the batteries:

-- Ten minutes charging the cover adds one hour battery life to the Kindle (per Amazon).
-- Only the Oasis has a USB port. The cover must be attached to the Oasis in order to be charged, it cannot be charged independently.
-- The Oasis is capable of 20 months total life if in hibernate mode (per Amazon).
-- It is possible to check the battery levels for both the cover and the Oasis itself, IF the Oasis is attached to the cover (see video and also photo appended to this review). After pressing the 'quick action' icon at the top of the screen, the display will show the battery level for the cover and Oasis separately and given as percentages. That's a very nice enhancement and none of the other Kindle models offer the ability to view battery percentage.
-- When charging there is a small amber LED that illuminates and it part of the on/off button. When fully charged it changes to green.

My first reaction after learning that the Oasis had a separate battery in the cover was to expect a very long battery life, but when I started looking at the actual specs I saw that's really not the case. My interpretation is that because Amazon has made light weight and thinness their top priorities (see 'Design Objectives' earlier in the review), they decided to forgo the opportunity to pack a huge battery into the cover. But perhaps that will be an option at some point in the future, since the Oasis ‘system’ is designed for the Oasis plus cover to work together and Amazon could easily offer a ‘high-capacity’ cover at some point in the future, for those who wanted such a thing and were willing to sacrifice some size and weight in order to get it.

WANT MORE INFORMATION?

Incredible as it may seem that anyone would desire more after working their way through this brevity-disabled review, there is also a very comprehensive Kindle Oasis Support page now available on Amazon, that has a great deal of information including video illustrations of various features and operations:
    """
    aspects = ["price", "design", "size", "display", "weight", "feel", "battery"]
    classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                          ('tfidf', TfidfTransformer(use_idf=True)),
                          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42,
                                                n_jobs=-1))])
    prepare_classifier(classifier)
    classify(reviews, aspects, classifier)
