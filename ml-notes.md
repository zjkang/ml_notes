https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018

```python
# save & load models using scikit-learn

from sklearn.linear_model import LogisticRegression
import pickle
model = LogisticRegression()
model.fit(xtrain, ytrain)
# save the model to disk
pickle.dump(model, open(model_file_path, 'wb'))

model = pickle.load(open(model_file_path, 'rb'))
result_val = model.score(xval, yval)
result_test = model.score(xtest, ytest)
```




## 11. subgradient descent



