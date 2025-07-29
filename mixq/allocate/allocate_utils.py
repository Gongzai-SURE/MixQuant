param_order = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ]

def modify_size(layer_sizes):
    # 将fisher格式变为转为list
    if isinstance(layer_sizes,dict):
        layer_sizes = list(layer_sizes.values())
        all_values = []
        for index, block in enumerate(layer_sizes):
            for key, value in block.items():
                all_values.append(value)
        return all_values
    else:
        return layer_sizes

def same_layer_reset(data):
    # 计算 layer 数量
    num_layers = len(data) // len(param_order)
    
    # 初始化结果字典
    result = {param: [] for param in param_order}
    
    # 按顺序填充数据
    for i in range(num_layers):
        start_idx = i * len(param_order)
        layer_data = data[start_idx : start_idx + len(param_order)]
        
        for param, value in zip(param_order, layer_data):
            result[param].append(value)

    return result

def pack_list(data):
    # 检查所有参数列表长度是否一致（即 layer 数量）
    num_layers = len(data[param_order[0]])
    assert all(len(data[param]) == num_layers for param in param_order), "参数长度不一致！"

    # 按 layer 顺序展开为一维向量
    new_data = []
    for layer_idx in range(num_layers):
        for param in param_order:
            new_data.append(data[param][layer_idx])

    return new_data


