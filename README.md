# AI-Education
캐글에서 실제 재해에 대한 트윗이 진짜 재해인지 아닌지를 예측하는 머신러닝모델 구현
[Link](https://www.kaggle.com/c/nlp-getting-started)
## Data set
### Files
* __train.csv__ - 훈련 세트
* __test.csv__ - 테스트 세트
* __sample_submission.csv__ - 샘플 제출파일

### Columns
* `id` - 각 트윗에 대한 고유 식별자
* `text` - 트윗의 텍스트
* `location` - 트윗에서 보낸 위치(비어있을 수 있음)
* `keyword` - 트윗의 특징 키워드(비어있을 수 있음)
* `target` - train.csv에만 있으며 트윗이 실제 재해(1)인지 아닌지(0)를 나타냄

## Code Execution Process
1. 데이터셋을 접근하고 확인
```python
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")

train.head()
```
　2.null값 존재 확인하고 결측값 대체
```python
train.isnull().sum(),test.isnull().sum()
train['location'] = train['location'].fillna(train['location'].mode()[0])
train['keyword'] = train['keyword'].fillna(train['keyword'].mode()[0])

test['location'] = test['location'].fillna(test['location'].mode()[0])
test['keyword'] = test['keyword'].fillna(test['keyword'].mode()[0])
```
　3.escape 된 문자열을 다시 원래 형태의 데이터 문자열로 변환
 ```python
train['processed_text'] = html.unescape(train['text'])
test['processed_text'] = html.unescape(test['text'])
 ```
 4. 불용어 제거, 어간 추출, 1번 초과로 나온 단어 제거
```python
stemmer = PorterStemmer()
words = stopwords.words("english")
train['processed_text'] = train['text'].apply(lambda x: " ".join([stemmer.stem(i)                                                              
                                                                    for i in re.sub("[^a-zA-Z]"," ",x).split() if i not in words]).lower())
test['processed_text'] = test['text'].apply(lambda x: " ".join([stemmer.stem(i)
                                                                  for i in re.sub("[^a-zA-Z]"," ",x).split() if i not in words]).lower())
                                                                  lists = train['processed_text'].str.split().tolist()
count = Counter(chain.from_iterable(lists))
train['processed_text'] = [' '.join([j for j in i if count[j]>1])for i in lists]

lists = test['processed_text'].str.split().tolist()
count = Counter(chain.from_iterable(lists))
test['processed_text'] = [' '.join([j for j in i if count[j]>1])for i in lists]
```
　5. 데이터 자르기
```python
y = train.target
X = train['processed_text']
X_test = test['processed_text']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = 0.10, shuffle = True, random_state = 42)
```
　6.TF-IDF를 이용한 자연어 특징 추출
 ```python
vectorizer_tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
train_tfIdf = vectorizer_tfidf.fit_transform(X_train.values.astype('U'))
val_tfIdf = vectorizer_tfidf.transform(X_val.values.astype('U'))
X_test_tfIdf = vectorizer_tfidf.transform(X_test.values.astype('U'))
```
　7.XGBoost를 이용한 모델링 및 F1 score 계산
 ```python
d_train = xgb.DMatrix(train_tfIdf, label=y_train)
watchlist = [(d_train, 'train')]
bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=20, verbose_eval=10)
predict_y = bst.predict(d_train)
print("The test log loss is:",log_loss(y_train, predict_y, eps=1e-15))
fpr_tfidf, tpr_tfidf, t_tfidf = roc_curve(y_train, predict_y)
print('F1 score',f1_score(y_train,predict_with_best_t(predict_y, find_best_threshold(t_tfidf,fpr_tfidf,tpr_tfidf)))) 
 ```
<a href="https://ibb.co/YPtJ01t"><img src="https://i.ibb.co/pb3NhM3/f1score.png" alt="f1score" border="0"></a>
## Result of Execution
비록 캐글에 제출한 코드의 정확성은 약 77%를 달성하였지만 정밀도와 재현율의 평균인F1 Score은 약 89%를 달성하였습니다.

<a href="https://ibb.co/bz4Ydtx"><img src="https://i.ibb.co/5BQCcZd/result.png" alt="result" border="0"></a>
 
