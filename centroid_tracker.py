# centroid_tracker.py

import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        """
        max_disappeared: 追跡中に顔が何フレーム連続で検出されなくても保持するか（閾値）。
        """
        self.nextObjectID = 0
        self.availableIDs = []
        self.objects = dict()       # objectID -> (centroid_x, centroid_y)
        self.disappeared = dict()   # objectID -> 連続で検出されなかったフレーム数
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """
        新しい顔を登録する
        """
        # self.objects[self.nextObjectID] = centroid
        # self.disappeared[self.nextObjectID] = 0
        # self.nextObjectID += 1
        if self.availableIDs:
            objectID = self.availableIDs.pop(0)
        else:
            objectID = self.nextObjectID
            self.nextObjectID += 1
        self.objects[objectID] = centroid
        self.disappeared[objectID] = 0

    def deregister(self, objectID):
        """
        追跡を解除する
        """
        del self.objects[objectID]
        del self.disappeared[objectID]
        self.availableIDs.append(objectID)
        self.availableIDs.sort()

    def update(self, rects):
        """
        rects: [(x, y, w, h), ...] のリスト
            MediaPipe Face Detection から得た矩形を pixel 座標で与える
        return: self.objects (objectID -> centroid)
        """
        # (1) もし矩形がひとつもなければ、すべての objectID を disappeared カウントする
        if len(rects) == 0:
            to_deregister = []
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    to_deregister.append(objectID)
            for objectID in to_deregister:
                self.deregister(objectID)
            return self.objects

        # (2) 各矩形から centroid を計算
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            input_centroids[i] = (cX, cY)

        # (3) 既存追跡中オブジェクトがない場合 → すべて新規登録
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # (4) 既存オブジェクトと新検出 centroid の距離行列を作成
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = distance.cdist(np.array(objectCentroids), input_centroids)

            # (5) 最小距離順にマッチング
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > 50:
                    # もし距離が 50px を超えていたら別人とみなす
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # (6) マッチしなかった既存顔 → disappeared カウントをインクリメント
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            # (7) マッチしなかった新顔 → 新規登録
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.register(input_centroids[col])

        return self.objects
