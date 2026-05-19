# Prediction_of_plasmids_destination

Title:** Predicting plasmid destinations: Harnessing k-mer compositions for conjugative transfer

DOI:** TBD

This page contains a script and dataset for predicting the destination of plasmids using the k-mer composition and DNABERT embedding described in Tokuda et al.

## Data availability

Large data files are not included in this GitHub repository.  
The datasets and trained models associated with this study are available from Figshare:

**Figshare DOI:** TBD

After publication, this section will be updated with the final Figshare DOI.

The Figshare dataset includes:
- experimental conjugation assay results
- 127 recipient metadata
- k-mer feature matrices for plasmid and chromosomes
- k-mer features matrices for prediction (canonical k-mer)
- DNABERT-derived feature matrices

## Overview of the analysis

This repository contains analysis scripts used in the manuscript:

**Tokuda et al. “***Predicting plasmid destinations: Harnessing k-mer compositions for conjugative transfer***”**

In this study, we developed sequence-based machine learning models to predict plasmid destinations during conjugative transfer. Plasmid destinations were defined based on one-on-one conjugation assay results, in which recipient strains that acquired a plasmid were treated as destination-positive and those with no detected plasmid acquisition were treated as destination-negative.

The computational analyses include:

1. construction of dendrogram based on  plasmid k-mer compositions;
2. calculation of k-mer composition features from genomes;
3. construction of plasmid–recipient feature matrices (k-mer compositions and DNABERT-derived embedding;
4. training and evaluation of random forest models;

Large input datasets, trained models, prediction outputs, and supplementary files are deposited on Figshare. 
This GitHub repository is intended to document the computational workflow and provide the scripts used for the analyses reported in the manuscript.


## **Dendrogram**

The dendrogram shown in FigureS2 was genarated using **Dendrogram.py**.
The input data for this script (plasmid fasta files) stored in plasmids_fasta directory were used. 

## **k-mer model**
### **k-mer_calculation**
The k-mer composiitons used for generating the violin plot in Figure 2B, as well as for features of machine learning, were calculated using **01_cal_kmers.py**. 
For features of machine learning, to use canonical k-mer, **01_cal_kmers_canonical.py** was used to calculate.
The input plasmid seqences are stored in k-mer_calculation directory, while 127 chromosomal sequences of recipient candidates have been uploaded in Figshare.
The output files correspond to "plasmid_kmer.csv" and "host_kmer.csv" for 14 plasmids and 127 recipient candidates, respectively. 

### Model_construction
**02_kmer_train_gridsearch.py** trains Random Frest models using k-mer composition differences as input features. Separated models are constructed for each k-mer lemgth from k = 2 to 7.
Model performance is ecaluated using nested cross-validation, and hyperparameters are optimized by grid search.
**02_2_eval_group.py** evaluates the trained k-mer models using group-based data splitting by plasmid or recipient. 
It is used to assess how well the models generalize to plasmids or recipients that were not included in the training data (Supplemental section 4).
**03_compare_voting.py** compares the prediction performance of individual k-mer models and the hard-voting ensemble model. 
Predictions from the k = 2 to k = 7 models are treated as votes, and performance metrics such as accuracy, F1 score, precision, recall, and AUC are compared while varying the number of positive votes required for a final positive prediction.

### Prediction
The predictions against 127 recipient candidates can be conducted for each k-mer using **04_predict_destination.py** by providing the target plasmid sequences as input.
For batch predictions involving multiple plasmids,**04_predict_folder.py** can be used execute all predictions at once.
The outputs include votes from individual k-mer models, total vote counts, vote fractions, and final predictions under different voting thresholds.
Due to their large size, the files "plasmid_kmer.csv" and "host_kmer.csv" have been uploaded to Figshare. 

## **DNABERT-derived embedding model**
### **DNABERT**
The features for DNABERT-derived embedding model were genarated by using **host_dnabert.py** and **plasmid_dnabert.py** from plasmids and chromosomes fasta.
Each genome sequence is split into 6-mer tokens and processed using the pretrained DNABERT model. 
The resulting 768-dimensional vectors are saved as `host_vectors.csv` and used as recipient-side features for DNABERT-based model training and prediction.

### Model_construction
The DNABERT-based prediction models were constructed using two main scripts. **02_train_dnabert.py** was used to train Random Forest models using DNABERT-derived sequence embeddings. 
Choromosome vectors and plasmid vectors were combined with plasmid–chromosome pair labels ("pair_converted.csv"), and three types of pairwise feature representations were generated: Difference (`Diff`), Absolute Difference (`AbsDiff`), and element-wise Product (`Prod`). 
A separate Random Forest model was trained for each representation. Model performance was evaluated using nested cross-validation, and hyperparameters were optimized by grid search.
**02_2_dnabert_eval_random_and_group.py** was used to evaluate the generalization performance of the DNABERT-based models. 
This script performed group-based cross-validation by grouping samples either by plasmid or by recipient. 
The plasmid-group split was used to evaluate prediction performance for plasmids not included in the training data, whereas the recipient-group split was used to assess prediction performance for recipients not represented in the training data.

### Prediction
The predictions against 127 recipient candidates can be conducted for each k-mer using **04_predict_dnabert_single.py** by providing the target plasmid sequences as input.
For batch predictions involving multiple plasmids,**04_predict_dnabert.py** can be used execute all predictions at once.
The script calculates a DNABERT embedding vector from the input plasmid sequence, combines it with precomputed recipient genome vectors, and applies the trained `Diff`, `AbsDiff`, and `Prod` models to each recipient. 
The output includes predicted probabilities and binary predictions for each DNABERT feature representation.


