# Text Data Processing Labs

Laboratory works for the university course **"Text Data Processing and Analysis"**, written in Python as Jupyter notebooks.

The eight labs follow a typical NLP learning path: from regular expressions and basic text preprocessing, through classical machine learning on text (vectorization, classification, topic modeling, sentiment analysis), to building the components of a task-oriented chatbot with spaCy and a fine-tuned transformer model.

> **Note:** code is in Python, while notebook comments and printed messages are in Ukrainian. All datasets and processed texts are in English.

## Contents

| Lab | Topic | Main tools |
|-----|-------|------------|
| [Lab 1](lab-1/lab-1.ipynb) | Regular expressions and basic text preprocessing | `re`, NLTK |
| [Lab 2](lab-2/lab-2.ipynb) | Text vectorization and document clustering | scikit-learn, SciPy, Gensim |
| [Lab 3](lab-3/lab-3.ipynb) | Text classification | scikit-learn, NLTK |
| [Lab 4](lab-4/lab-4.ipynb) | Topic modeling and collocations | Gensim, NLTK |
| [Lab 5](lab-5/lab-5.ipynb) | Sentiment analysis | spaCy, scikit-learn, TextBlob |
| [Lab 6](lab-6/lab-6.ipynb) | Custom NER and intent classification | spaCy |
| [Lab 7](lab-7/lab-7.ipynb) | Rule-based matching and dependency parsing | spaCy |
| [Lab 8](lab-8/lab-8.ipynb) | Fine-tuning a transformer for dialogue responses | Transformers, PyTorch |

## Repository structure

```
text-data-processing-labs/
├── lab-1/   lab-1.ipynb, text9.txt
├── lab-2/   lab-2.ipynb, doc9.txt
├── lab-3/   lab-3.ipynb, ecommerceDataset4.csv
├── lab-4/   lab-4.ipynb, IT_tickets2.csv
├── lab-5/   lab-5.ipynb, movie3.csv
├── lab-6/   lab-6.ipynb, examples.json, service_model/, intent_model/
├── lab-7/   lab-7.ipynb, homes.json
└── lab-8/   lab-8.ipynb, homes.json
```

Each folder is self-contained: the notebook sits next to the data it reads, and the notebooks are committed with their outputs, so the results can be viewed directly on GitHub without running anything.

## Labs

### Lab 1. Regular expressions and basic text preprocessing

Works with a short English text (`text9.txt`).

- Finds all house numbers in the text with a regular expression and writes them back reversed.
- Counts the words in the text, ignoring punctuation and special characters.
- Tags parts of speech and extracts the verbs.
- Removes stop words.
- Stems the words of the fifth sentence with the Porter stemmer.

### Lab 2. Text vectorization and document clustering

Works with a small noisy corpus (`doc9.txt`) that mixes documents on different topics.

- Cleans and lemmatizes the corpus.
- Builds a **bag-of-words** representation and extracts the vector of a given word.
- Builds a **TF-IDF** representation.
- Groups the documents with **hierarchical agglomerative clustering** (Ward linkage) and plots a dendrogram.
- Trains a **Word2Vec** model, prints the vectors of selected words and finds the words most similar to them.

### Lab 3. Text classification

Classifies e-commerce product descriptions by category (`ecommerceDataset4.csv`).

- Preprocesses the texts (cleaning, stop-word removal, POS-aware lemmatization) and balances the classes by sampling.
- Represents the texts as bigrams in two ways: **bag of bigrams** and **TF-IDF**.
- Trains two classifiers on each representation: **Multinomial Naive Bayes** and **Linear SVM**.
- Tunes hyperparameters with `GridSearchCV` and compares the models by accuracy, classification report and confusion matrix.

### Lab 4. Topic modeling and collocations

**Part 1** runs topic modeling on IT support tickets (`IT_tickets2.csv`).

- Trains **LSI** models with 2 to 10 topics and picks the best number of topics by the C_v coherence score.
- Shows the terms that describe each topic.
- Finds the documents that contribute most to each topic.
- Assigns topics to three new, unseen documents.

**Part 2** finds the top 10 **trigram collocations** by PMI in `chesterton-brown.txt` from the NLTK Gutenberg corpus.

### Lab 5. Sentiment analysis

Predicts the sentiment of movie reviews (`movie3.csv`, binary labels).

- Preprocesses the reviews with spaCy (lemmatization, stop-word removal).
- Trains a **logistic regression** classifier on TF-IDF features and evaluates it with accuracy and a confusion matrix.
- Classifies the same reviews with the lexicon-based **TextBlob** polarity score.
- Compares both approaches, including on three randomly chosen reviews.

### Lab 6. Custom NER and intent classification

The first step towards a home rental chatbot.

- **Named entity recognition:** fine-tunes the NER component of `en_core_web_sm` to recognise a custom `Service` entity (for example "two-bedroom apartment" or "schedule a visit") using hand-labelled examples from `examples.json`. The trained pipeline is saved in `service_model/`.
- **Intent classification:** trains a spaCy `textcat_multilabel` component on user utterances from the home rental dialogues to predict intents such as `FindApartment` and `ScheduleVisit`. The trained pipeline is saved in `intent_model/`.

> The intent classifier reads `homes.json`, which is not stored in `lab-6/`. Copy it from `lab-7/` or `lab-8/` before re-running that part.

### Lab 7. Rule-based matching and dependency parsing

Extracts structured information from user utterances in the home rental dialogues (`homes.json`).

- Uses the spaCy `Matcher` with token patterns to detect the requested **number of bedrooms** and user **confirmations** ("yes", "sounds good", "that works" and so on), skipping matches that are nested inside longer ones.
- Uses the **dependency parse** to extract verb and direct object pairs (for example "find" + "apartment"), which reveal what the user wants to do.

### Lab 8. Fine-tuning a transformer for dialogue responses

Builds the response generator of the home rental chatbot.

- Turns the dialogues from `homes.json` into pairs of user utterance and system response.
- Fine-tunes **BART** (`facebook/bart-base`) on these pairs for 10 epochs with the encoder frozen.
- Saves the model and generates replies for a sample conversation: asking for the area and number of bedrooms, answering a question about pets, and scheduling a visit.

## Getting started

The notebooks were created with **Python 3.12**.

1. Clone the repository:

   ```bash
   git clone https://github.com/lilvload/text-data-processing-labs.git
   cd text-data-processing-labs
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate

   pip install jupyter numpy pandas matplotlib scipy scikit-learn nltk gensim spacy textblob torch transformers
   ```

3. Download the language resources:

   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt punkt_tab stopwords wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng gutenberg
   ```

4. Start Jupyter from the folder of the lab you want to run, because the notebooks load their data by relative path:

   ```bash
   cd lab-1
   jupyter notebook lab-1.ipynb
   ```

### Notes on running

- **Lab 8** downloads `facebook/bart-base` from the Hugging Face Hub on first run. Training works on a CPU but is much faster on a GPU. The fine-tuned model is saved locally to `my_home_rental_model/` and is not included in the repository.
- **Lab 6** already includes the trained spaCy pipelines, so the test cells can be run without retraining.
- Some results (train/test splits, model training, random samples) are not seeded, so re-running a notebook can give slightly different numbers from the saved outputs.

## Technologies

Python · Jupyter Notebook · NLTK · spaCy · scikit-learn · Gensim · TextBlob · Hugging Face Transformers · PyTorch · pandas · NumPy · SciPy · Matplotlib
