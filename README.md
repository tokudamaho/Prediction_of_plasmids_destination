# Prediction_of_plasmids_destination

This page contains a script for predicting the destination of plasmids using the k-mer composition described in Tokuda et al.

**Dendrogram**

Figure S1で作成したデンドログラムは，Dendrogram.pyのスクリプトを使用して描いた．
本スクリプトでのinputデータ（chr126は，Figshareに格納した．
プラスミド塩基配列はplasmids_fastaディレクトリ内のファスタファイルを使用した．
出力ファイルはFigshareの＊＊＊に該当する．


**k-mer_calculation**

Figure 2B，S2のバイオリン図および機械学習の特徴量として使用した，k-mer頻度の算出には，k-merfreq.both.pyを使用した．入力したプラスミド配列はk-mer_calculationディレクトリ内に，染色体塩基配列（ドラフトゲノムを含む）はFigshareに格納した．
出力ファイルはFigshareの＊＊＊に該当する．

**violin_histogram**

Figure 2を描いたスクリプト
Figure 2Aはhistogram.py, Figure 2B, S2はviolin.pyを用いて描いた．
Figure 2B, S2にはk-mer_calculationによって出力したファイル＊＊＊を入力した．

