import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# importing all libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings("ignore")

NER_FEATURES = ['CARDINAL',
 'DATE',
 'EVENT',
 'FAC',
 'GPE',
 'LANGUAGE',
 'LAW',
 'LOC',
 'MONEY',
 'NORP',
 'ORDINAL',
 'ORG',
 'PERCENT',
 'PERSON',
 'PRODUCT',
 'QUANTITY',
 'TIME',
 'WORK_OF_ART']

POS_FEATURES = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

LINGUISTIC_CAT_FEATURES = ['sentence_count_cat', 'word_count_cat', 'words_per_sentence_cat', 'average_word_length_cat', 'large_words_cat']

LINGUISTIC_NUM_FEATURES = ['sentence_count', 'word_count', 'words_per_sentence', 'average_word_length', 'large_words']

CONDENSED_LINGUISTIC_FEATURES = ['pron_words_ratio', 'pron_sents_ratio', 'adj_sents_ratio',	'adj_words_ratio']

MATH_CAT_FEATURES = ['has_exp', 'has_mod', 'has_logarithm', 'has_fraction', 'has_eq', 'has_neq', 'has_pow', 'has_symbol', 'has_digits']

MATH_NUM_FEATURES = ['no_of_exps', 'no_of_pow', 'symbol_count' ,'mod_count', 'log_count', 'fracs_count', 'eqlts_count', 'neqlts_count', 'max_degree_of_equations', 'number_of_digits', 'number_of_numbers']

MANDATORY_FEATURES = ['no_of_equations', 'no_of_variables', 'type']

MATH_VOCAB_FEATURES = ['number_of_math_vocab']

TARGET_FEATURE = ['level']

GPT_TARGET_FEATURE = ['gpt_val']

def plot_confusion_matrix(confusion_matrix, labels):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # Set labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

    # Set colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set title and labels
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Show the plot
    plt.show()

def evaluation(y_train_pred, y_test_pred, y_train, y_test):
  print("train_evaluation:\n")
  print(classification_report(y_train_pred, y_train))
  print(confusion_matrix(y_train_pred, y_train))
  print("\ntest_evaluation:\n")
  print(classification_report(y_test_pred, y_test))
  print(confusion_matrix(y_test_pred, y_test))
  labels = ["Level 1", "Level 2", "Level 3"]
  if y_train.nunique() == 5:
      labels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
  plot_confusion_matrix(confusion_matrix(y_test_pred, y_test), labels)


# Level1 + Level2 -> Level1, Level3->Level2, Level4+Level5 -> Level3
def club_class(class_var):
    if class_var == 'Level 1' or class_var == 'Level 2':
        return 'Level 1'
    elif class_var == 'Level 3' or class_var == 'Level 4':
        return 'Level 2'
    else:
        return 'Level 3'

def encode_target(class_var):
    return int(class_var.split(" ")[1])


#definition of error metrics function
scores_df = pd.DataFrame(columns=['Model','F1_train','F1_test'])
def get_metrics(train_act,train_pred,test_act,test_pred,model_description,dataframe):
    F1_train = f1_score(train_act,train_pred, average='weighted')
    F1_test = f1_score(test_act, test_pred, average='weighted')
    s1=pd.Series([model_description,F1_train,F1_test],
                                           index=scores_df.columns)
    dataframe = dataframe.append(s1, ignore_index=True)
    return(dataframe)


def rf_model(data, test_size = 0.2, use_smote_technique=1, target_feature="level", club_target=False, experiment="Experiment", scores=scores_df):
    
    data1 = data

    # Seperate the target variable 
    X = data1.drop(columns = [target_feature])
    y = data1[target_feature]

    if y.dtype != "int64":
        y = y.apply(encode_target)

    print(y.dtypes)

    # Split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 2, stratify = y)
    if club_target:
        y_train = y_train.apply(club_class)
        y_test = y_test.apply(club_class)

    # Smote the data
    if use_smote_technique == 1:
        # Count the class distribution before applying SMOTE
        print("Class distribution before SMOTE:", Counter(y_train))

        # Apply SMOTE to the dataset
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Count the class distribution after applying SMOTE
        print("Class distribution after SMOTE:", Counter(y_train))

    # Random Forest Classifier - Machine Learning Model
    rfc=RandomForestClassifier(n_jobs=-1, random_state=42)
    param_grid = {
        'max_depth': [5, 7, 9],                      # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],             # Minimum number of samples required to split a node
        'max_features': ['auto'],    # Number of features to consider at each split
        'criterion': ['gini', 'entropy', 'log_loss'],
        'oob_score': [True],
        'n_estimators': [25],
    }

    if use_smote_technique != 1:
        param_grid['class_weight'] = ['balanced', 'balanced_subsample'] 

    # Perform grid search to find the best combination of parameters
    grid_search = GridSearchCV(rfc, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Obtain the best values and best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    # Evaluation of model
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    y_train_pred = grid_search.best_estimator_.predict(X_train)
    evaluation(y_train_pred, y_test_pred, y_train, y_test)

    # obtain Best Features
    best_model = grid_search.best_estimator_
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    scores = get_metrics(y_train, best_model.predict(X_train), y_test, best_model.predict(X_test), experiment, scores)

    print("Feature Importance Rankings:")
    for i, feature in enumerate(X.columns[indices]):
        print(f"{i + 1}. {feature}: {importances[indices[i]] * 100}")
    
    return scores