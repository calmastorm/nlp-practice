#!/bin/bash
#SBATCH --partition=M1                    # 指定分区
#SBATCH --qos=q_d8_norm                   # 指定 QoS
#SBATCH --nodes=1                         # 使用的节点数量
#SBATCH --gres=gpu:1                      # 请求 1 个 GPU
#SBATCH --cpus-per-task=4                 # 请求 2 个 CPU
#SBATCH --time=6:00:00                   # 设置最大运行时间为 4 小时
#SBATCH --mem=16G                         # 设置内存大小为 8 GB
#SBATCH --job-name=ML                    # 设置作业名称
#SBATCH --output=output_%x_%j.out         # 设置标准输出文件名
#SBATCH --error=error_%x_%j.err           # 设置错误日志文件名

# 加载必要的模块和环境
module load anaconda                      # load anaconda
eval "$(conda shell.bash hook)"
conda activate msai_env             # 激活 msai_env Conda 环境

# 运行 Python 脚本
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.33
# export LD_PRELOAD=/home/msai/hu0023an/.conda/envs/msai_env/lib/libstdc++.so.6.0.33
# export LD_LIBRARY_PATH=$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:$LD_LIBRARY_PATH
python test/test.py
