from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value,LpStatus

# 创建问题实例
prob = LpProblem("LP_Problem", LpMinimize)

# 定义变量
x1 = LpVariable("x1", lowBound=0)
x2 = LpVariable("x2", lowBound=0)
x3 = LpVariable("x3", lowBound=0)
x4 = LpVariable("x4", lowBound=0)
x5 = LpVariable("x5", lowBound=0)
x6 = LpVariable("x6", lowBound=0)

# 目标函数
prob += 8 * x1 + 5.3 * x2 + 3 * x3 + 2 * x4 + 1.5 * x5

# 约束条件
prob += x1 + x2 + x3 + x4 + x5 + x6 == 1
prob += 2 * x1 + 3 * x2 + 4 * x3 + 6 * x4 + 8 * x5 + 16 * x6 <= 4.2

# 求解问题
prob.solve()

# 输出结果
print("Status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Objective =", value(prob.objective))