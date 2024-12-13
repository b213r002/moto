import numpy as np

class ReversiProcessor:
    def process_tournament(self, df):
        # 試合が切り替わる盤面リセット
        if df["tournamentId"] != self.now_tournament_id:
            self.table_info = [0] * 100  # 10x10のボードで外枠は無視する
            # 初期配置
            self.table_info[44] = 2
            self.table_info[45] = 1
            self.table_info[54] = 1
            self.table_info[55] = 2
            self.turn_color = 1
            self.now_tournament_id = df["tournamentId"]
        else:
            self.turn_color = 1 if self.turn_color == 2 else 2

        # 置ける箇所がなければパスする
        if len(self.GetCanPutPos(self.turn_color, self.table_info)) == 0:
            self.turn_color = 1 if self.turn_color == 2 else 2

        # 配置場所
        put_pos = df["move"]

        # 訓練用データ追加
        self.record_training_data(put_pos)

        # 盤面更新
        put_index = put_pos[0] + (put_pos[1]) * 10
        self.PutStone(put_index, self.turn_color, self.table_info)

    def record_training_data(self, put_pos):
        # ボード情報を自分と敵のものに分ける
        my_board_info = np.zeros(shape=(8, 8), dtype="int8")
        enemy_board_info = np.zeros(shape=(8, 8), dtype="int8")

        for i in range(11, 89):  # 10x10のボードの内側(8x8部分)を処理
            if i % 10 == 0 or i % 10 == 9:
                continue  # 余分な枠をスキップ

            board_x = (i % 10) - 1
            board_y = (i // 10) - 1

            if self.table_info[i] == 1:
                my_board_info[board_y][board_x] = 1
            elif self.table_info[i] == 2:
                enemy_board_info[board_y][board_x] = 1

        move_one_hot = np.zeros(shape=(8, 8), dtype='int8')
        move_one_hot[put_pos[1]][put_pos[0]] = 1

        # 訓練データを記録
        if self.turn_color == 1:
            self.my_board_infos.append(np.array([my_board_info.copy(), enemy_board_info.copy()], dtype="int8"))
            self.my_put_pos.append(move_one_hot)
        else:
            self.enemy_board_infos.append(np.array([enemy_board_info.copy(), my_board_info.copy()], dtype="int8"))
            self.enemy_put_pos.append(move_one_hot)

    # ダミーのメソッド(詳細な実装が必要)
    def GetCanPutPos(self, turn_color, table_info):
        # 置ける場所をリストとして返す
        # 実際にはこの部分でルールに従った処理が必要
        return [pos for pos in range(100) if table_info[pos] == 0]

    def PutStone(self, put_index, turn_color, table_info):
        # 石を置く処理
        # 実際にはこの部分でルールに従った処理が必要
        table_info[put_index] = turn_color

