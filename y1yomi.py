import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import time

# CSVから試合内容を読み込む
def load_match_info(self):
    # csv読み込み
    csv_data = pd.read_csv("wthor.csv")
 
    # 1行目はヘッダになっているため削除する
    # 試合内容はtranscriptの列
    csv_data  =  csv_data.drop(index= csv_data[csv_data["transcript"].str.contains("transcript")].index)
 
    # 正規表現を使って2文字ずつ切り出す
    extract_one_hand = csv_data["transcript"].str.extractall('(..)')
 
    # Indexを再構成して、1行1手の表にする
    # 試合の切り替わり判定のためtournamentIdも残しておく
    one_hand_df =  extract_one_hand.reset_index().rename(columns={"level_0":"tournamentId" , "match":"move_no", 0:"move_str"})
         
    # アルファベットを数字に変換するテーブル
    conv_table = {"a" : 1, "b" : 2, "c" : 3, "d" : 4, "e" : 5, "f" : 6, "g" : 7, "h" : 8}
    one_hand_df["move"] = one_hand_df.apply(lambda x: self.convert_move(x["move_str"], conv_table), axis=1)
 
    return one_hand_df
 
# 1手を数値に変換する
def convert_move(self, v, conv_table):
    l = conv_table[v[:1]] # 列の値を変換する
    r = int(v[1:]) # 行の値を変換する
    return np.array([l - 1, r - 1], dtype='int8')

# def process_tournament(self, df):
#     # 試合が切り替わる盤面リセット
#     if df["tournamentId"] != self.now_tournament_id:
#         self.table_info = [0] * 100
#         self.table_info[44] = 2
#         self.table_info[45] = 1
#         self.table_info[54] = 1
#         self.table_info[55] = 2
#         self.turn_color = 1
#         self.now_tournament_id = df["tournamentId"]
#     else:
#         self.turn_color = 1 if self.turn_color == 2 else 2
 
#     # 置ける箇所がなければパスする
#     if len(reversi.GetCanPutPos(self.turn_color, self.table_info)) == 0:
#         self.turn_color = 1 if self.turn_color == 2 else 2
     
    
#     put_pos = df["move"]
 
# # 訓練用データ追加
#     self.record_training_data(put_pos)
 
#     # 盤面更新
#     put_index = put_pos[0] + 1 + (put_pos[1] + 1) * 10
#     reversi.PutStone(put_index, self.turn_color, self.table_info)
 
