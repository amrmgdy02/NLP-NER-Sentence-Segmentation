## 🍕 NLP-Based Pizza & Drink Order Parser  

  
## 📌 Project Overview
This project develops an NLP system that converts natural language food orders into a structured format. It uses BERT-based Named Entity Recognition (NER) to extract relevant details like pizza toppings, sizes, crust types, and drinks.

## Features
✅ Named Entity Recognition (NER) with BERT for extracting structured information  
✅ Entity Parsing for Pizza & Drink Orders  
✅ Machine Learning & Deep Learning Models (LSTM, Transformer-based models)  
✅ JSON Output for Easy Integration with Ordering Systems  
✅ Competitive Ranking on Kaggle  

## Project Pipeline
1️⃣ Data Preprocessing → Cleaning, Tokenization  
2️⃣ Feature Extraction → BOW, TF-IDF, Word Embeddings  
3️⃣ Model Training → Transformer-based NER model (dslim/bert-ner)  
4️⃣ Entity Extraction & JSON Parsing  
5️⃣ Final Testing & Evaluation  

## Technologies Used
Hugging Face Transformers (BERT-based NER Model: dslim/bert-ner)  
Pandas & NumPy for data handling

## Installation

### Prerequisites
- NUMPY (only if you want to train the model)  
- Pandas (only if you want to train the model)  
- Scikit-learn (only if you want to train the model)  
- Transformers  
- JSON
  
### Steps
1. **Clone the repository:**
2. **Download the training set from here => https://www.kaggle.com/datasets/amrrmagdy10/pizza-train**
3. **Download Models from here: 1) NER model => https://drive.google.com/drive/folders/1KzQWvt6dw9nARdNoEJ89lOxwgsw9egXo | 2) IS model => https://drive.google.com/drive/folders/1D_Ev-j1m1qRawQPUINFOl0Yu7VcwBqJX**
4. **Inside FineTuning-dslim-bert-ner.ipynb file, in the first cell, change dataset_path with the your downloaded dataset actual path**
5. **Inside ready-bert-ner.ipynb file, in the second cell, update path for models and tokenizers with tour actual path "model and tokenizer will have the same path each model (NER or IS)"**
6. **Run all**
   **Note that step 2&4 are needed only if you want to train the model yourself, but you can test the model only with steps 1,3,5 and 6**

## 🎯 Goal: Build a highly accurate NLP model for structured food order processing!
