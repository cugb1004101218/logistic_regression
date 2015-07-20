# -*- coding: utf-8 -*-

# 梯度下降的步长
step_len = 0.1

# 迭代次数
iteration_num = 100

# 训练数据文件路径
# 格式：
# 每一行是一个样本，feature之间用tab分割，末尾1表示正样本，0表示负样本。
# 例如：巧克力\t包邮\t日本\t进口\1
train_data_file_path = './meishi_small.txt'

# 训练后生成的模型路径
output_model_file_path = './model.txt'
