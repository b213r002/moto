import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import time

# MatchLoader: WTHORから試合データを読み込む
class MatchLoader:
    def __init__(self, csv_file):
        self.csv_file = csv_file
    
    def load_match_info(self):
        """CSVファイルから試合内容を読み込む"""
        try:
            csv_data = pd.read_csv(self.csv_file, header=None)
            move_sequences = csv_data.iloc[:, -1]
            extract_one_hand = move_sequences.str.extractall(r'(..)')
            # 中間結果を新しいCSVファイルに保存
            move_sequences.to_csv('result1.csv', index=False)

            one_hand_df = extract_one_hand.reset_index().rename(
                columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"}
            )

            conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
            one_hand_df["move"] = one_hand_df["move_str"].apply(lambda x: self.convert_move(x, conv_table))
            
            return one_hand_df
        except Exception as e:
            print(f"試合情報の読み込み中にエラーが発生しました: {e}")
            return None

    def convert_move(self, move_str, conv_table):
        """1手を数値に変換する"""
        col = conv_table[move_str[0]]
        row = int(move_str[1])
        return np.array([col - 1, row - 1], dtype='int8')

# ReversiProcessor: オセロの進行を処理
class ReversiProcessor:
    def __init__(self):
        self.table_info = np.zeros(100, dtype='int8')  # 10x10のボード
        self.my_board_infos = []
        self.enemy_board_infos = []
        self.my_put_pos = []
        self.enemy_put_pos = []
        self.turn_color = 1
        self.now_tournament_id = None

    def process_tournament(self, df):
        """試合の進行を処理"""
        if df["tournamentId"] != self.now_tournament_id:
            self.reset_board()
            self.now_tournament_id = df["tournamentId"]
        else:
            self.turn_color = 1 if self.turn_color == 2 else 2

        if not self.GetCanPutPos(self.turn_color):
            self.turn_color = 1 if self.turn_color == 2 else 2

        put_pos = df["move"]
        self.record_training_data(put_pos)
        self.PutStone(put_pos)

    def reset_board(self):
        """盤面をリセットする"""
        self.table_info.fill(0)
        self.table_info[44] = 2
        self.table_info[45] = 1
        self.table_info[54] = 1
        self.table_info[55] = 2

    def record_training_data(self, put_pos):
        """訓練用データを記録"""
        my_board_info = np.zeros((8, 8), dtype="int8")
        enemy_board_info = np.zeros((8, 8), dtype="int8")

        for i in range(11, 89):
            if i % 10 in {0, 9}:
                continue
            board_x = (i % 10) - 1
            board_y = (i // 10) - 1

            if self.table_info[i] == 1:
                my_board_info[board_y][board_x] = 1
            elif self.table_info[i] == 2:
                enemy_board_info[board_y][board_x] = 1

        move_one_hot = np.zeros((8, 8), dtype='int8')
        # put_posの値を確認し、範囲外の値でないことを確認
        assert 0 <= put_pos[0] <= 7 and 0 <= put_pos[1] <= 7, "put_posの値が範囲外です"
        move_one_hot[put_pos[1]][put_pos[0]] = 1

        if self.turn_color == 1:
            self.my_board_infos.append(np.array([my_board_info, enemy_board_info], dtype="int8"))
            self.my_put_pos.append(move_one_hot)
        else:
            self.enemy_board_infos.append(np.array([enemy_board_info, my_board_info], dtype="int8"))
            self.enemy_put_pos.append(move_one_hot)

    def GetCanPutPos(self, turn_color):
        """置ける場所をリストとして返す"""
        return [pos for pos in range(100) if self.table_info[pos] == 0]

    def PutStone(self, put_pos):
        """石を置く処理"""
        put_index = put_pos[0] + put_pos[1] * 10
        self.table_info[put_index] = self.turn_color

class ReversiModel:
    def __init__(self, my_board_infos, enemy_board_infos, my_put_pos, enemy_put_pos):
        self.my_board_infos = my_board_infos
        self.enemy_board_infos = enemy_board_infos
        self.my_put_pos = my_put_pos
        self.enemy_put_pos = enemy_put_pos
        self.model = self.create_model()

        # 記録直後にデータの形状を確認する
        if len(my_board_infos) > 0:
            print(f"my_board_info shape: {my_board_infos[0].shape}")
            print(f"enemy_board_info shape: {enemy_board_infos[0].shape}")

    def create_model(self):
        """ニューラルネットワークモデルの作成"""
        class Bias(layers.Layer):
            def __init__(self, input_shape):
                super(Bias, self).__init__()
                self.W = tf.Variable(initial_value=tf.zeros(input_shape), trainable=True)

            def call(self, inputs):
                return inputs + self.W

        model = keras.Sequential()
        model.add(layers.Permute((2, 3, 1), input_shape=(2, 8, 8)))
        for _ in range(7):
            model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
            model.add(layers.Conv2D(1, kernel_size=1, use_bias=False))
            model.add(layers.Flatten())
            model.add(Bias((64,)))
            model.add(layers.Activation('softmax'))

            model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        print('モデルは正常に作成されました。')
        return model

    def training(self):
        """モデルの訓練"""
        x_train = np.concatenate([self.my_board_infos, self.enemy_board_infos])
        y_train_tmp = np.concatenate([self.my_put_pos, self.enemy_put_pos])
        y_train = y_train_tmp.reshape(-1, 64)


        # Tensor Boardコールバックの設定
        tb_cb = keras.callbacks.TensorBoard(log_dir='model_log/relu_12', histogram_freq=1, write_graph=True)

        start_time = time.time()  # 訓練開始時間を記録

        try:
            self.model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.2, callbacks=[tb_cb])
        
        except KeyboardInterrupt:
            self.model.save('saved_model_reversi/my_model_interrupt')
            print('中断しました。モデルを保存しました。')
            return

        end_time = time.time()  # 訓練終了時間を記録
        training_time = end_time - start_time  # 訓練にかかった時間を計算
        print(f'訓練が完了しました。モデルを保存しました。訓練時間: {training_time:.2f}秒')

        self.model.save('saved_model_reversi/my_model')

# ランダムAI
class ReversiRandomAI:
    def __init__(self, reversi_processor):
        self.processor = reversi_processor
    
    def select_random_move(self):
        """ランダムな場所に石を置く"""
        possible_moves = self.processor.GetCanPutPos(self.processor.turn_color)
        if possible_moves:
            random_move = random.choice(possible_moves)
            return np.array([random_move % 10 - 1, random_move // 10 - 1], dtype='int8')
        return None

# モデル vs ランダムAI の対戦ロジック
class ReversiGame:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.random_ai = ReversiRandomAI(processor)
    
    def predict_next_move(self):
        """モデルが次の手を予測する"""
        # 最後のボード情報を取得
        my_board_info = self.processor.my_board_infos[-1]
        enemy_board_info = self.processor.enemy_board_infos[-1]

        # デバッグ用出力
        print("my_board_info.shape:", my_board_info.shape)
        print("enemy_board_info.shape:", enemy_board_info.shape)

        # x_inputの作成
        x_input = np.array([my_board_info, enemy_board_info])
        print("x_input before reshape:", x_input.shape)

        # x_inputをリシェイプ
        # ここで、x_inputの要素数に合わせて形状を変更する
        # 例: x_inputの要素数が512の場合
        x_input = np.reshape(x_input, (2, 2, 8, 8))  # バッチサイズを2にする

        print("x_input after reshape:", x_input.shape)

        # モデルで予測
        predictions = self.model.predict(x_input)
        move_index = np.argmax(predictions)
        return np.array([move_index % 8, move_index // 8], dtype='int8')
    
    def play_game(self):
        """モデルとランダムAIが対戦する"""
        self.processor.reset_board()  # 盤面をリセット
        turn = 1  # 1がモデル、2がランダムAI

        while True:
            if turn == 1:
                # モデルが次の手を予測して置く
                move = self.predict_next_move()
            else:
                # ランダムAIが次の手を選んで置く
                move = self.random_ai.select_random_move()
            
            if move is not None:
                self.processor.record_training_data(move)  # 訓練データに記録
                self.processor.PutStone(move)  # 石を置く
            else:
                # 置ける場所がない場合、パスする
                print(f"ターン {turn}: 置ける場所がないためパスします")
                if not self.processor.GetCanPutPos(1 if turn == 2 else 2):
                    print("ゲーム終了")
                    break
            
            # 盤面の表示（デバッグ用）
            self.print_board()
            
            # ターン交代
            turn = 1 if turn == 2 else 2

        self.determine_winner()

    def print_board(self):
        """盤面を表示する"""
        board_str = ""
        for y in range(1, 9):
            for x in range(1, 9):
                piece = self.processor.table_info[y * 10 + x]
                if piece == 1:
                    board_str += "X "  # モデルの石
                elif piece == 2:
                    board_str += "O "  # ランダムAIの石
                else:
                    board_str += ". "
            board_str += "\n"
        print(board_str)

    def determine_winner(self):
        """勝者を判定して表示"""
        model_score = np.sum(self.processor.table_info == 1)
        random_ai_score = np.sum(self.processor.table_info == 2)
        print(f"モデルの得点: {model_score}, ランダムAIの得点: {random_ai_score}")
        if model_score > random_ai_score:
            print("モデルの勝利！")
        elif model_score < random_ai_score:
            print("ランダムAIの勝利!")
        else:
            print("引き分け！")


if __name__ == "__main__":
    match_loader = MatchLoader("a.csv")
    one_hand_df = match_loader.load_match_info()

    reversi_processor = ReversiProcessor()
    for _, row in one_hand_df.iterrows():
        reversi_processor.process_tournament(row)

    # Processorからモデルにトレーニングデータを渡す
    reversi_model = ReversiModel(
        reversi_processor.my_board_infos,
        reversi_processor.enemy_board_infos,
        reversi_processor.my_put_pos,
        reversi_processor.enemy_put_pos
    )

    # モデルを訓練
    reversi_model.training()

    # 対戦を開始
    game = ReversiGame(reversi_model.model, reversi_processor)
    game.play_game()