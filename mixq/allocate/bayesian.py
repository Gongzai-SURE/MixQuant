import optuna
from optuna.trial import TrialState
import numpy as np
import json
import warnings
from functools import partial
import time

class BayesianOptimization:
    def __init__(self, bits, p, F, B, R, alpha, n_trials=8000):
        self.bits = bits
        self.p = p
        self.F = F
        self.B = B
        self.P_total = np.sum(p)
        self.R = R
        self.alpha = alpha
        self.n_trials = n_trials
        self.best_trial = None
        self.study = optuna.create_study(directions=["minimize", "minimize"])

    def objective(self, trial):
        bit_allocation = [trial.suggest_categorical(f"bit_{i}", self.bits) for i in range(len(self.F))]
        bit_allocation = np.array(bit_allocation)
        
        # 目标1：敏感度加权和
        objective_value = np.sum(self.F * np.exp(-self.alpha * (self.B / bit_allocation)))
        
        # 目标2：压缩率约束违反程度
        compressed_size = np.sum(self.p * (bit_allocation / self.B))
        constraint_violation = np.abs(compressed_size - self.P_total * self.R)
        
        return objective_value, constraint_violation

    def print_intermediate_results(self, study, trial):
        if trial.state != TrialState.COMPLETE:
            return
        
        if trial.number % 1000 == 0:
            all_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            
            feasible_trials = [t for t in all_trials if t.values[1] <= 1e-3]
            if len(feasible_trials) > 0:
                best_trial = min(feasible_trials, key=lambda t: t.values[0])
                is_feasible = True
            else:
                best_trial = min(all_trials, key=lambda t: t.values[0])
                is_feasible = False
            
            print(f"\n=== 轮次 {trial.number} ===")
            print(f"当前最优敏感度加权和: {best_trial.values[0]:.4f}")
            # print(f"压缩率约束违反程度: {best_trial.values[1]:.4f}")
            print(f"是否满足约束: {is_feasible}")
            print(f"位宽分配: {[best_trial.params[f'bit_{i}'] for i in range(len(self.F))]}")

    def select_best_trial_with_fallback(self, study, constraint_threshold=1e-3):
        all_trials = study.trials
        
        feasible_trials = [
            t for t in all_trials
            if t.values[1] <= constraint_threshold and t.state == optuna.trial.TrialState.COMPLETE
        ]
        
        if len(feasible_trials) > 0:
            best_feasible = min(feasible_trials, key=lambda t: t.values[0])
            print("找到满足约束的解")
            allocation_res = np.array([best_feasible.params[f"bit_{i}"] for i in range(len(self.F))])
            return allocation_res
        else:
            best_overall = min(all_trials, key=lambda t: t.values[0])
            warnings.warn("未找到满足约束的解，返回目标1最优的解")
            allocation_res = np.array([best_overall.params[f"bit_{i}"] for i in range(len(self.F))])
            return allocation_res

    def run(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(partial(self.objective), 
                            n_trials=self.n_trials, 
                            callbacks=[self.print_intermediate_results],
                            show_progress_bar=True)
        self.best_trial = self.select_best_trial_with_fallback(self.study)

    def get_best_trial(self):
        return self.best_trial




# 测试代码
"""     def compute_compression(self):
        bit_allocation =  self.get_best_trial()
        return np.sum(self.p * (bit_allocation / self.B)) / self.P_total
    
    @staticmethod
    def load_json(file):
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data = []
        for id, block in enumerate(json_data):
            for key, value in json_data[block].items():
                data.append(value)
        return np.array(data)


# 示例用法
if __name__ == "__main__":
    bits = [3, 4, 8]  
    F = BayesianOptimization.load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/fisher_data.json')
    p = BayesianOptimization.load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/LayersParams.json')
    B = 16 
    R = 0.25
    alpha = 0.01  #量化位宽影响系数
    t_start = time.time()
    bayesian_opt = BayesianOptimization(bits, p, F, B, R, alpha)
    bayesian_opt.run()
    print("最优解：",bayesian_opt.get_best_trial())
    print("压缩率:", bayesian_opt.compute_compression())
    t = round((time.time() - t_start),1)
    print(f"Running Time : {t} s") """