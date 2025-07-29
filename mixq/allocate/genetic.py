import random
import numpy as np
from deap import base, creator, tools
import json
from .allocate_utils import *

class GeneticAlgorithm:
    def __init__(self, bits, p, F, B, R, alpha, 
                 population_size=10000, max_generations=50, cxpb=0.7, mutpb=0.2, elite_size=20):
        self.bits = bits
        self.p = p
        self.F = F
        self.B = B
        self.P_total = sum(p)
        self.R = R
        self.alpha = alpha
        self.population_size = population_size
        self.max_generations = max_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.elite_size = elite_size
        self.N = len(p)  # 个体长度
        self.best_individual = None

        # 定义适应度函数和个体
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # 初始化工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bit", random.choice, self.bits)  # 每个基因从 bits 中选择
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)  # 两点交叉
        self.toolbox.register("mutate", self.mutate_individual, indpb=0.1)  # 自定义变异
        self.toolbox.register("select", tools.selTournament, tournsize=10)  # 竞标赛选择

    def get_best_individual(self):
        return self.best_individual

    def create_individual(self):
        return tools.initRepeat(creator.Individual, self.toolbox.attr_bit, n=self.N)

    def repair_individual(self, individual):
        """
        Directed repair based on sensitivity, so that the individual strictly meets the constraints.
        Bit width adjustment is strictly limited to bits = [2, 4, 8].
        If repair is not possible, the original individual is returned and marked as unrepairable.
        """
        def get_current_sum(ind):
            return sum(self.p[i] * (ind[i] / self.B) for i in range(self.N))

        target_sum = self.P_total * self.R
        current_sum = get_current_sum(individual)
        max_attempts = 3  # 最大修复尝试次数
        attempt = 0

        while abs(current_sum - target_sum) > 1e-6 and attempt < max_attempts:
            if current_sum > target_sum:
                # 压缩率高于目标值：降低某层的位宽
                candidate_indices = [i for i in range(self.N) if individual[i] > min(self.bits)]
                if not candidate_indices:
                    break  # 没有可调整的层
                target_idx = min(candidate_indices, key=lambda i: self.F[i])
                # 将该层的位宽调整到下一个更低位宽
                current_bit = individual[target_idx]
                current_bit_index = self.bits.index(current_bit)
                new_bit = self.bits[current_bit_index - 1] if current_bit_index > 0 else current_bit
            else:
                # 压缩率低于目标值：提高某层的位宽
                candidate_indices = [i for i in range(self.N) if individual[i] < max(self.bits)]
                if not candidate_indices:
                    break  # 没有可调整的层
                target_idx = max(candidate_indices, key=lambda i: self.F[i])
                # 将该层的位宽调整到下一个更高位宽
                current_bit = individual[target_idx]
                current_bit_index = self.bits.index(current_bit)
                new_bit = self.bits[current_bit_index + 1] if current_bit_index < len(self.bits) - 1 else current_bit

            # 更新位宽
            individual[target_idx] = new_bit
            current_sum = get_current_sum(individual)
            attempt += 1

        # 检查是否修复成功
        if abs(current_sum - target_sum) <= 1e-6:
            return individual, True  # 修复成功
        else:
            return individual, False  # 修复失败

    def evaluate_individual(self, individual):
        """评估个体的适应度"""
        repaired_individual, is_repaired = self.repair_individual(individual)

        if is_repaired:
            # 如果修复成功，则计算目标函数值
            objective = sum(self.F[i] * (np.exp(-self.alpha * (repaired_individual[i]/self.B))- np.exp(-self.alpha)) / (np.exp(-self.alpha * (1.5/self.B))- np.exp(-self.alpha)) \
                           for i in range(self.N))
            return objective,
        else:
            # 如果修复失败，则添加高代价惩罚项
            constraint = sum(self.p[i] * (individual[i] / self.B) for i in range(self.N)) - self.P_total * self.R
            penalty = 1e6 * (constraint ** 2)  # 高代价惩罚项
            return 1e6 + penalty,  # 返回一个较大的值表示不可行解

    def mutate_individual(self, individual, indpb):
        """对个体进行变异"""
        for i in range(len(individual)):
            if random.random() < indpb:  # 以 indpb 的概率进行变异
                # 从 bits 中随机选择一个不同于当前值的位宽
                new_bit = random.choice([bit for bit in self.bits if bit != individual[i]])
                individual[i] = new_bit
        return individual,

    def initialize_population(self):
        """初始化种群"""
        tmp_best_individual = [int(self.R * self.B)] * self.N
        population = self.toolbox.population(n=self.population_size)
        # 将 best_individual 转换为 Individual 类型并加入种群
        best_ind = creator.Individual(tmp_best_individual)
        population[0]=best_ind
        return population
    
    def eaWithElitism(self, population, toolbox, cxpb, mutpb, ngen, elite_size=10, verbose=True):
        logbook = tools.Logbook()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        
        # 初始种群评估
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(ngen):
            # 选择下一代
            offspring = toolbox.select(population, len(population) - elite_size)
            offspring = list(map(toolbox.clone, offspring))
            
            # 交叉和变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 保留精英
            elite = tools.selBest(population, elite_size)
            offspring.extend(elite)
            
            # 更新种群
            population[:] = offspring
            
            # 记录统计信息
            logbook.record(gen=gen, **stats.compile(population))
            if verbose:
                print(logbook.stream)
        
        return population, logbook

    def run(self):
        """运行遗传算法"""
        population = self.initialize_population()
        results, log = self.eaWithElitism(
            population, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.max_generations, elite_size=self.elite_size, verbose=True
        )
        self.best_individual = tools.selBest(results, k=1)[0]
        return results, log


 # 测试代码
if __name__ == "__main__":
    def load_json(file):
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data = []
        for id, block in enumerate(json_data):
            for key, value in json_data[block].items():
                data.append(value)
        return np.array(data)

    # 定义参数
    bits = [2,3,4]  # 可选位宽
    F = load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/fisher_data.json')
    p = load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/LayersParams.json')
    N = len(F)  # 层数
    B = 16  # 原始位宽
    P_total = np.sum(p)  # 总参数规模
    R = 0.1875  # 压缩率 0.25 0.1875
    alpha = 10  # 目标函数中的超参数

    # 创建遗传算法实例
    ga = GeneticAlgorithm(bits, p, F, B, R, alpha)

    # 运行遗传算法
    result, log = ga.run()

    # 输出最优解
    best_individual = tools.selBest(result, k=1)[0]
    best_fitness = best_individual.fitness.values[0]
    print("Best individual:", best_individual)
    print("fisher:", F)
    print("Best fitness:", best_fitness) 
    

