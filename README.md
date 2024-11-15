# Prediction_of_plasmids_destination

This page contains a script for predicting the destination of plasmids using the k-mer composition described in Tokuda et al.

**Dendrogram**

Figure S1で作成したデンドログラムは，Dendrogram.pyのスクリプトを使用して描いた．
本スクリプトでのinputデータ（chr126は，Figshareに格納した．
プラスミド塩基配列はplasmids_fastaディレクトリ内のファスタファイルを使用した．


**k-mer_calculation**

Figure 2Bのバイオリン図および機械学習の特徴量として使用した，k-mer頻度の算出には，k-merfreq.both.pyを使用した．入力したプラスミド配列はk-mer_calculationディレクトリ内に，染色体塩基配列（ドラフトゲノムを含む）はFigshareに格納した．

