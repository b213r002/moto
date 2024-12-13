import pandas as pd
import numpy as np

# CSVファイルの読み込み
df = pd.read_csv("W.csv", header=None)

# 最後の列（ムーブシーケンス）だけを抽出
move_sequences = df.iloc[:, -1]  # 最後の列を抽出
# 中間結果を新しいCSVファイルに保存
move_sequences.to_csv('result.csv', index=False)
# 正規表現を使って2文字ずつ切り出す
transcripts_raw = move_sequences.str.extractall(r'(..)')  # 正規表現で2文字ずつ分割

# Indexを再構成して、1行1手の表にする
transcripts_df = transcripts_raw.reset_index().rename(columns={"level_0": "game_id", "level_1": "move_no", 0: "move_str"})

# 列の値を数字に変換するdictionaryを作る
def left_build_conv_table():
    left_table = ["a", "b", "c", "d", "e", "f", "g", "h"]
    left_conv_table = {}
    n = 1

    for t in left_table:
        left_conv_table[t] = n
        n += 1

    return left_conv_table

left_conv_table = left_build_conv_table()

# dictionaryを使って列の値を数字に変換する
def left_convert_column_str(col_str):
    return left_conv_table[col_str]  

# 1手を数値に変換する関数
def convert_move(v):
    l = left_convert_column_str(v[:1])  # 列の値を変換
    r = int(v[1:])  # 行の値を変換
    return np.array([l - 1, r - 1], dtype='int8')  # 配列で返す

# 全ての手を数値に変換
transcripts_df["move"] = transcripts_df.apply(lambda x: convert_move(x["move_str"]), axis=1)

# 数値に変換されたムーブを確認
print(transcripts_df.head())

# 新しいCSVファイルとして保存
transcripts_df.to_csv("converted_moves.csv", index=False)
