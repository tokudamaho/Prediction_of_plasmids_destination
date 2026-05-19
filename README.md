# Prediction_of_plasmids_destination

Title:** TBD
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

## **phylogenetic analysis**


## **k-mer_calculation**

The k-mer composiitons used for generating the violin plot in Figure 2B and Figure S2, as well as for features of machine learning, were calculated using **k-merfreq_both.py**. 
The input plasmid seqences are stored in k-mer_calculation directory, while 127 chromosomal sequences of recipient candidates have been uploaded in Figshare.
The output files correspond to "plasmidk_noth.csv" and "chr127_k_dra_both.csv" for 14 plasmids and 127 recipient candidates, respectively. 

## **destination_prediction**

The prediction of plasmid destinations was conducted in three step: **Model_construstion**, **Cross-validation_test** and **Prediction**. 

### Model_construction
Using the results of **k-merfreq_both.py**, the differences in k-mer compositions between 14 plasmids and 127 bacterial chromsomes recipient candidates were calculated. Then, **data_preparation3.py** (input and output files should be designed in the script) was used to combine the two csv files with the results of conjugation assays (**labels2.csv**) to create matrices for each k-mer. 
To tune the model parameters, the **grid_search.py** was executes. 
usage: python grid_search.py <input_file data_with_labelsk2.csv> <outpit_directory> 

### Repeated Hold-Out Validation



### Prediction
predictions against 127 recipient candidates can be conducted for each k-mer using **predictionk_tune.py** by providing the target plasmid sequences as input.
usage: predictionk_tune.py <input_plasmidfasta_file> <output_directry>
These scripts were integrated in **predict_main.py**, which outputs rge final prediction results based on majirity voting for k = 2~8.
usage: python predict_main.py <input_plasmidfasta_file> <output_directry> <final_output_file>
For batch predictions involving multiple plasmids,** main_folder.py** can be used execute all predictions at once.
usage: python main_folder.py <input_plasmid_fasta_diretory> <output_directoty>

Due to their large size, the files "plasmidk_noth.csv", "chr127_k_dra_both.csv" and "data_with_labelsk2.csv" have been uploaded to Figshare. 


