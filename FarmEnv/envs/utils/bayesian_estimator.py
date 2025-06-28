import numpy as np
from collections import defaultdict, deque

from FarmEnv.envs.config import MAX_LENGTH
class BayesianEstimator:
    def __init__(self):

        self.history = defaultdict(lambda: deque(maxlen=MAX_LENGTH))

    def update(self, grow_stage, weather_s, allocated_res, grown):
        self.history[(grow_stage, weather_s)].append({
            "allocated_res": allocated_res,
            "grown": grown
        })

    def estimate(self, grow_stage, weather_s):
        records = self.history.get((grow_stage, weather_s), [])
        if len(records) < 5:
            return 10.0

        #Wether grow success or not
        grown_allocate = [r["allocated_res"] for r in records if r["grown"]]
        not_grown_allocate = [r["allocated_res"] for r in records if not r["grown"]]

        if not grown_allocate or not not_grown_allocate:
            return np.mean([r["allocated_res"] for r in records])

            # 估计阈值：介于“最小成功”与“最大失败”之间
        min_grow = min(grown_allocate)
        max_fail = max(not_grown_allocate)

        # 返回中间值 + 加一点扰动
        estimate = (min_grow + max_fail) / 2.0
        return float(np.random.normal(loc=estimate, scale=0.1))

