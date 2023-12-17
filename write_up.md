## Introduction {#introduction .unnumbered}

In a world of continuous media and information, many of us rarely read
over all of the article descriptions, let alone every article in their
entirety. The chance that we will may be increased if we find the topic
is something that we are curious about. The keyword here is \"topic.\"
How are we supposed to know what the topic is without reading the
article? This is where the keywords of an article become extremely
important. These words need to have unique role of summarizing and
categorizing the article. However, a writer cannot subscribe an article
with an unlimited amount of words (i.e. the article itself). It has been
established that using Machine Learning techniques, we are able to
distinguish when two words are similar in meaning, so what about a whole
bunch of words? Can a computer do this with just the description? For a
model to be good at this task it must be able to determine at least one
of the keywords used and identify when a keyword is not being used for a
given article. Since the model is dealing with the meaning of similar
words, I predict that the models that associate more featrues together
will do better.

## Data {#data .unnumbered}

The data set I used was imported from Kaggle and created by the user
Pedro Araujo Ribeiro (2023). It contained a list of 4,112 CNN articles
from 2023. Ribeiro simply web-scraped the data from the site and broke
the data up into seven columns, each representing a different part of
the articles ( ID, Title, Description, Body, Keywords, Theme, Link). For
this project, I will be focusing on just the Description and Keywords
columns. Preliminary processing of the data yielded 3464 unique
keywords, which were too many labels for my model to predict, so I
decided to focus on the five most frequent ones (' domestic alerts', '
international alerts', ' continents and regions', ' brand safety-nsf
sensitive', ' iab-business and finance').

::: {#table:1}
  Description                                                                                                                                                                                                                                                                                      Keywords
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  South Korean President Yoon Suk Yeol warned on Wednesday that his country and its allies "will not stand idly by" if North Korea receives Russian help to boost its weapons of mass destruction -- just days after the leaders of the two nuclear-armed nations held a closely watched summit.   \['asia', **'brand safety-nsf sensitive'**, 'brand safety-nsf war and military', 'brand safety-nsf weapons', **'continents and regions'**, **'domestic alerts'**, 'domestic-international news', 'east asia', 'eastern europe', 'europe', 'government organizations - intl', 'military', 'military weapons', 'north korea', 'north korea nuclear development', 'nuclear weapons', 'political figures - intl', 'russia', 'russia-ukraine conflict', 'south korea', 'united nations', 'unrest', 'conflicts and war', 'vladimir putin', 'weapons and arms', 'weapons of mass destruction', 'yoon suk-yeol'\]

  : This is an example of one of the article's description and list of
  keywords. Bolded are the 5 most common keyword occurrences.
:::

Since not all of the rows included these five keywords, the data needed
to be further processes by determining whether to keep. Each row entry.
The description for each row is tokenized using Natural Language Toolkit
(NLTK) and counted for unigrams, bigrams, and trigrams and entered into
a dictionarys For each row, I create a length five vectors following
one-hot encoding, where each entry representing 1 if the keyword was
applied to it and 0 if not. If a given row did not include one of the
five keywords, a TypeError would be raised, and the row was excluded.
There were some cases where the description was empty; these rows were
also excluded. As seen in the example given in Table
[1](#table:1){reference-type="ref" reference="table:1"}, an entry will
not contain all of the keywords and the keywords that aren't include in
the five most common list are not included as well.

The list of the feature dictionaries are then vectorized using
DictVectorizer and leaving sparse = True. By doing so, the data can be
split into train/dev/test sets with an approximate 80, 10, 10 split. In
the end there were 2,271 in train, 310 in dev, and 287 in test. with a
total of 2,868 entries.

## Dev Set Results {#dev-set-results .unnumbered}

As previously mentioned, my model uses one-hot encoding for the labels,
so I chose Logisitic Regression and Random Forest as the estimator for
MultiOutputClassifier to generate binary classification for each label.
The main hyperparameters I tuned are solver/C and max_features/max_n,
respectively. To tune the models I looped through each solver in
\[\"lbfgs\", \"liblinear\", \"newton-cg\", \"sag\", \"saga\"\] with C in
\[100, 10, 1.0, 0.1, 0.01\] and each max_features in \[10, 100, 1000\]
with max_n in \[1, 2, 3, \... 20\]. Since this is 25 and 60
configurations respectively, I have only presented the top performers in
Table [2](#table:2){reference-type="ref" reference="table:2"}. As seen
in the table the optimized hyperparameters for the features were the
same.

To determine the performance of each configuration, the models were
judge on an accuracy metric. The accuracy_score method in sklearn
yielded subpar results because this metric determines if the predicted
labels is identical to the true labels (\"exact-match\" accuracy).
Instead, accuracy is determine on whether least one of the keywords is
identified (\"one-match\" included accuracy). This significantly
increased the accuracy. Although I did not tune for this, I noticed that
the model had a hard time not predicting the keywords, shown in the
accuracy scores based on if the model predicted a keyword being excluded
from the vector (\"one-match\" excluded accuracy). Increasing the number
of features to be predicted I found that increasing the number of
features increases this second accuracy, but lowers the sklearn
accuracy_score and increases the processing time.

::: {#table:2}
  --------------------- ---------- ---------------- -----------
  Model                 Features   Hyperparametes   Accuracy
                        Unigram    saga/100         95.82
                        Bigram     saga/100         **97.21**
  Logistic Regression   Trigram    saga/100         95.47
                        Unigram    1000/20          96.17
                        Bigram     1000/20          94.77
  Random Forest         Trigram    1000/20          **96.86**
  --------------------- ---------- ---------------- -----------

  : The best preforming hyperparameters from the dev set. For Logisitic
  Regression, the hyperparameters are represented as solver/C and, for
  Random Forest, the hyperparameters are represented as
  max_features/max_n.
:::

## Test Set Results {#test-set-results .unnumbered}

The test set results were less dependent of the given models despite
having differing results in the Dev Set. As shown in Table
[3](#table:3){reference-type="ref" reference="table:3"}, in both models,
the tests using unigram features had the best results. As mentioned
before, it is important that the model does not over predict labels as
well. This criteria is test for by using the second metric of accuracy,
mentioned in the Dev Set Results section. As seen in Table
[\[tabel:4\]](#tabel:4){reference-type="ref" reference="tabel:4"},
unigrams still outperform the other features.

::: {#table:3}
  Model                 Features   Hyperparametes   Accuracy (\"one-match\" included)   Accuracy (\"one-match\" excluded)   Accuracy (\"exact-match\")
  --------------------- ---------- ---------------- ----------------------------------- ----------------------------------- -----------------------------
                        Unigram    saga/100         **97.91**                           68.64                               **40.42**
                        Bigram     saga/100         96.17                               **69.34**                           **37.63**
  Logistic Regression   Trigram    saga/100         95.12                               64.11                               34.15
                        Unigram    1000/20          **98.26**                           65.51                               **38.33**
                        Bigram     1000/20          95.82                               **63.41**                           32.4
  Random Forest         Trigram    1000/20          96.17                               61.32                               29.27

  : This is the results of using the same hyperparameters from the best
  dev test configuration. The accuracy is using all three metrics. For
  Logistic Regression, the hyperparameters are represented as solver/C
  and, for Random Forest, the hyperparameters are represented as
  max_features/max_n.
:::

::: {#table:4}
  Model                 Features   Hyperparameters   Accuracy (\"one-match\" included)   Accuracy (\"one-match\" excluded)   Accuracy (\"exact-match\")
  --------------------- ---------- ----------------- ----------------------------------- ----------------------------------- ----------------------------
                        Unigram    saga/100          95.82                               99.65                               **12.89**
                        Bigram     saga/100          96.17                               **100.0**                           **12.89**
  Logistic Regression   Trigram    saga/100          **96.86**                           **100.0**                           11.85
                        Unigram    1000/20           **96.86**                           99.65                               **12.54**
                        Bigram     1000/20           96.52                               **100.0**                           8.71
  Random Forest         Trigram    1000/20           **96.86**                           **100.0**                           8.36

  : This table is the model using 20 features intsead of 20. The models
  use the same hyperparameters from the best dev test configuration. The
  accuracy is all three metrics. For Logistic Regression, the
  hyperparameters are represented as solver/C and, for Random Forest,
  the hyperparameters are represented as max_features/max_n.
:::

## Discussion {#discussion .unnumbered}

Although the different models didn't behave that differently, the
differing features did. This suggests that the frequency of each feature
in the training data has is more important than giving contextual
information. This is surprising to me given that I tend to use context
clues when determining the theme of a text. However, the model is
choosing from a finite set of choices, where the meaning behind the
individual words do not particularly matter. The results when
experimenting with the number of features suggest that the keywords that
I chose might have overlapped too much, creating a too ambiguous label
set. Yet, this configuration did decrease the \"exact match\" accuracy,
without changing the \"one-match\" accuracy. That means that by
increasing the number of labels increases the sparsity and the model has
to be more cautious. That being said, the model performed well with
identifying at least one of the keywords.

## Conclusion {#conclusion .unnumbered}

In summary, this project explores how well a machine learning model can
predict which keywords are going to be used with a given article. The
model was tasked to predict from a set of five most common keyword
labels. Instead of using the built in accuracy_score method, which
judges accuracy based on identical matches, accuracy worked best when
judging based on if true labels and prediction labels matched at least
once. The model did well predicting at least one of the keywords, but
with a low number of labels to predict it was very generous with its
predictions. It would be interesting to see how it would perform with a
more carefully chosen five labels, optimizing sparsity and whether it
can could generate on top of predict out of a finite set of keywords.