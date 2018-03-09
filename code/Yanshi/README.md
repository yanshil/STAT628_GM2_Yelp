## Scripts

* run_text2vec.py: Run Doc2vec Input ML model

* run_skML.py: Run Machine Learning Models of sklearn with tidied input

## Packages

* extra_features2csr.py: Tidy Non-Text/Category features to csr sparse matrix
	
```
# Example Usage
import extra_features2csr
get_extra_features(data, text_length=True, num_upper_words=True,
                       num_exclamation_mark=True, city=True,
                       num_question_mark=True, num_dollar=True,
                       num_percent=True, num_facebad=True, num_facegood=True)
```

* process_text.py: Define Text processing rules
	* decontracted
	* remove_punctuation
	* process_reviews
	* get_category_sp
	* count_upper_word

* sklearn_pack.py: Do Model training with large sparse matrix with sklearn.
	* Random Forest
	* Decision Tree
	* Neural Network

```
# Example Usage
import sklearn_pack

final_RF_pred = random_forest(finalX_train2, trainDF.stars, finalX_test2, n_parallel=4)
final_RF_pred.to_csv('predict_RF', index=False)
```

## Util

Some tiny scripts to do the explaratory analysis

## chtc

Submit templates for running high-memory jobs in [CHTC](http://chtc.cs.wisc.edu/)
