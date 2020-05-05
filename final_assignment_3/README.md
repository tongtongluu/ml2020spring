# Assignment 3: Clustering

### Machine Learning, PSAM 5020, Spring 2020 

### 0. Topic: [Option 2](https://github.com/visualizedata/ml/tree/master/ML_assignment_3/option_2)

 use [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) to cluster a set of images **on their metadata.** The measure of success is subjective; you know you have chosen the right features and number of clusters when the images in each cluster "seem" like they belong together. 

### 1. Problem: Overfitting

Generally, the model seems already overfitting. I was shocked at the crunch of dots at the top left corner. And hesitated if i should choose some model that has large space to improve. 

The overall challenge that I'm trying to think through is that the models are already quite overfit to the dataset – using the source code I'm getting over 99% accuracy and very high precision and recall scores for most of the models. However, when applied to my test data, these high scores do not translate, meaning that our models are likely overfit. I tried making a lot of adjustments to to the bag of words feature extraction, as well as adding additional features using regular expressions and new symbol counts for exclammation marks and question marks, but was having trouble getting the accuracy, precision and recall to improve on the test data. 


### 2. Solution

#### 2.1 High false positives or High false negatives
在feature选择中，加入了 “！” ， 
Here I added three new features, and ended up removing the two feautres that were in the starter code. 

First, I added a count of exclammation points for each comment. Next I added a count of uppercase words (two or more uppercase letters adjacent to each other) using a regular expression. Finally, I added a column with counts of three or more question marks grouped together in a comment.

``` python
# create regex for uppercase words 
    uppercase_pattern = '([A-Z]){2,}'

# create additional quantitative features
    # features from Amazon.csv to add to feature set
    toxic_data['exclam_count'] = toxic_data['comment_text'].str.count("\!")
    toxic_data['uppercase'] = len(re.findall(uppercase_pattern, str(toxic_data['comment_text'])))   
    toxic_data['questions'] = len(re.findall('(\?){3,}', str(toxic_data['comment_text'])))

    X_quant_features = toxic_data[["exclam_count", "uppercase", "questions"]]

```

#### 2.2 Bags of words: one word, two words, or up to seven words
Here I decided to change the hashing vectorizer to a [count vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) so that I could use n-grams or word groupings, instead of just individual words. I tried different numbers of word groupings from starting with 1-3 and up to 5. I also tried using a tfidf vectorizer (which combines the count vectorizer with tfidf transformer), however I got better results when using the count vectorizer with the tfidf transformer. 

``` python
count_vectorizer = CountVectorizer(max_features=10000)
    X_cv = count_vectorizer.fit_transform(toxic_data.comment_text)
    
    count_vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(3, 5), max_features=200000)
    X_cv2 = count_vectorizer2.fit_transform(toxic_data.comment_text)
    
    transformer = TfidfTransformer(smooth_idf=False)
    X_tfidf = transformer.fit_transform(X_cv)
```
#### 2.3 随便改别的参数
HashingVectorizer(n_features=2 ** 19,ngram_range=(1, 5),norm='l1',analyzer='word',stop_words="english",tokenizer=LemmaTokenizer(),alternate_sign=False)

Numerical features were also added, namely count of exclamation marks, question marks and uppercase characters. The ridge regression model was found to perform best, with parameters alpha=5 and normalize=True. The model is here.

### 3. Selected MODEL: RGD
也改了RGD的参数
As mentioned above, I chose this model because it had the lowest false negative rate. I'll need to make sure I can decrease the false positive rate using this model, or else I'll need to explore other models.

Results on the training data:
``` python
{'Pos': 13010, 'Neg': 114646, 'TP': 12930, 'TN': 102781, 'FP': 11865, 'FN': 80, 'Accuracy': 0.9064282133233064, 'Precision': 0.5214761040532365, 'Recall': 0.9938508839354343, 'desc': 'rdg_train'}
```

Results on the test data:
``` python
{'Pos': 3215, 'Neg': 28700, 'TP': 2976, 'TN': 23661, 'FP': 5039, 'FN': 239, 'Accuracy': 0.834623217922607, 'Precision': 0.37130380536494073, 'Recall': 0.9256609642301711, 'desc': 'rdg_test'}
```
Overall ROC plots for all models on test and training data:

Training:
直接把截图加到文件夹，改png的部分

![training data](roc_train.png)

Test:

![test data](roc_test.png)




### 4. Thoughts

we want a model that has high rate of correctly identified positive classifications, and a low rate of false negatives (toxic comments that aren't flagged as toxic). However, we also want to reduce the amount of false positives so that non-toxic comments aren't over censored. 就是要牺牲一些准确率，宁愿predict more positives



