# Prediction_of_plasmids_destination

This page contains a script for predicting the destination of plasmids using the k-mer composition described in Tokuda et al.

##**Dendrogram**

The dendrogram shown in FigureS1 was genarated using **Dendrogram.py**.
The input data for this script (plasmid fasta files) stored in plasmids_fasta directory were used. 

##**k-mer_calculation**

The k-mer composiitons used for generating the violin plot in Figure 2B and Figure S2, as well as for features of machine learning, were calculated using **k-merfreq_both.py**. 
The input plasmid seqences are stored in k-mer_calculation directory, while 127 chromosomal sequences of recipient candidates have been uploaded in Figshare.
The output files correspond to "plasmidk_noth.csv" and "chr127_k_dra_both.csv" for 14 plasmids and 127 recipient candidates, respectively. 

##**destination_prediction**

The prediction of plasmid destinations was conducted in three step: **Model_construstion**, **Cross-validation_test** and **Prediction**. 

plasmid destinationの予測は，Model_construction, Cross-validation test, destination_predictionの3つのステップで行った．k-merfreq_both.pyによって算出した"plasmidk_noth.csv"および，"chr127_k_dra_both.csv"よりそれぞれ14種類のプラスミド，127種類のrecipient candidatesのdifferences in k-mer compositionsを算出した．
その後，data_preparation3.py (input, outputファイルは，script上で指定），上記の2 filesと，results of conjugation assays (labels2.csv) を結合した行列をkごとに作成した．
モデルパラメーターのチューニングのため，grid_search.pyを実行した．
usage: python grid_search.py <input_file data_with_labelsk2.csv> <outpit_directory> 
同定された最適なパラメーターを採用したpredictionk_tune.pyによって各k-merごとに127株のrecipient candidatesに対する予測を実行可能である．
usage: predictionk_tune.py <input_plasmidfasta_file> <output_directry>
これらのスクリプトはpredict_main.pyでまとめて実行し，k=2~8での結果を多数決したものを最終予測結果として出力する．
usage: python predict_main.py <input_plasmidfasta_file> <output_directry> <final_output_file>
予測する対象プラスミドが，複数の場合は，main_folder.pyを使用して，一度に予測を実行する．
usage: python main_folder.py <input_plasmid_fasta_diretory> <output_directoty>

"plasmidk_noth.csv", "chr127_k_dra_both.csv" and "data_with_labelsk2.csv"はデータ量が膨大であるため，Figshareに格納した．


