# Natural Language Processing Assignments

This repository contains my solutions for the **Natural Language Processing (NLP) course** that I attended during the first semester - academic year 2021/2022 - of **M.Sc. in Machine Learning &amp; Big Data** at the **University of Naples _"Parthenope"_** (@uniparthenope).
The course covered a wide range of NLP topics, and I completed three assignments, which are described below:

## Assignment 1: Text Normalization using NLTK Library

For the first assignment, I prepared Jupiter notebook for explaining text normalization using the Natural Language Toolkit (NLTK) library. The notebooks covered the following operations:

- **Corpus loading**: The first step was to load a corpus of text data to be normalized.

- **Corpus statistics**: I performed some basic exploratory data analysis to understand the corpus, including word frequency analysis and visualization.

- **Tokenization**: I used NLTK's tokenization functionality to break down the text into individual words or tokens.

- **Lemmatization**: I used NLTK's WordNetLemmatizer to reduce each word in the corpus to its base or root form, which can help with standardization and normalization.

- **Stemming**: I also implemented stemming using NLTK's PorterStemmer, which is the process of reducing words to their base form by removing the suffix.

The notebooks were designed to provide a clear and thorough explanation of the text normalization process using NLTK.

## Assignment 2: Sentiment Analysis using Naive Bayes Classifier and Logistic Regression

For the second assignment, I used various NLP techniques to perform sentiment analysis on a dataset of movie reviews. Specifically, I used the NLTK library to preprocess the text data and prepare it for modeling. I also utilized TF-IDF vectorization and Continuous Bag of Words (CBOW) from the gensim library to vectorize word tokens. I then trained and evaluated a Naive Bayes classifier and logistic regression model to predict the sentiment of each review.

This was the provided script:
 * **Prepare Jupiter notebooks for explaining Sentiment Analysis with Naïve Bayes and Logistic regression**
 * Also, consider any preprocessing step
 * It would be possible to use any python library for NLP e for Machine Learning, Data Analysis, and numerical computation (e.g., scikit-learn, Pandas, and NumPy)

## Assignment 3: Auto-Complete using N-Gram Language Model

For the third assignment, I implemented an auto-complete system using an N-gram language model. I used the NLTK library to preprocess a dataset of text data and generate N-grams of varying lengths. I then used the N-grams to build a probabilistic language model that could predict the next word in a sentence. Finally, I implemented an auto-complete function that would suggest the most likely completion of a user's partial sentence based on the language model.

This was the provided script.
* Prepare Jupiter Notebooks for explaining Auto-Complete operation using N-Gram Language Models:

1. **Load and preprocess data**
    * Load and tokenize data.
    * Split the sentences into train and test sets.
    * Replace words with a low frequency by an unknown marker <unk>.
2. **Develop N-gram based language models**
    * Compute the count of n-grams from a given data set.
    * Estimate the conditional probability of a next word with k-smoothing.
3. **Evaluate the N-gram models by computing the perplexity score.**
4. **Use your own model to suggest an upcoming word given your sentence.**

## Technologies Used
The assignments were completed using Python and various NLP libraries, including:

- _**NLTK**_: A popular Python library for NLP tasks, including text normalization.
- _**Scikit-learn**_: A machine learning library used for sentiment analysis.
- _**Gensim**_: A library used for topic modeling, similarity detection, and text summarization.
- _**Numpy**_: A library used for numerical operations, including language modeling.
- _**Pandas**_: A library used for data manipulation and analysis.
- _**Matplotlib**_: A library used for data visualization.
- _**Seaborn**_: A library built on top of Matplotlib, used for data visualization.

## Getting Started
To run the code in this repository, you will need to have Python and the required libraries installed on your machine. You can also run the notebooks directly on **Google Colab**, where all of the assignments were completed. 
Anyway, all the assignments are self-explainable, so they contain all the needed instructions for reproducing the experiments. 

## License
This project is licensed under the **MIT License**.

Feel free to modify, distribute, and use the code in this repository for personal or commercial purposes. However, please attribute the original source and maintain the same license.



 
