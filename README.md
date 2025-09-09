# 3D Molecular Diffusion

基于扩散模型的3D分子生成项目。


## 环境要求

- Python 3.8+
- PyTorch 2.5.1+
- CUDA 12.4+

## 1. 环境配置

### 创建python3.8虚拟环境
```
conda create -n py38 python=3.8
```
### 启动虚拟环境
```
conda activate py38
```

## 2. 安装依赖

### 必要的安装python库

```
pip install -r requirements.txt
```

### 安装PyTorch Geometric相关包

**注意**：请确保PyTorch Geometric的版本与您的PyTorch和CUDA版本匹配。如果不匹配，请修改上述命令中的版本号。此处使用pytorch2.5.1对应在AutoDL的RTX4090中可选配置

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

## 3. 使用模型

### 训练模型(已自带权重文件，可以直接生成分子，也可自行重新训练)

```
python scripts/train_drug3d.py 
```

### 生成分子

```
python scripts/sample_drug3d.py 
```

### 评估结果

```
python scripts/evaluate_all.py 
```
## 配置说明

- `configs/train_MolDiff_simple.yml`: 训练配置
- `configs/sample_MolDiff_simple.yml`: 采样配置

## 注意事项

1. 确保PyTorch Geometric版本与PyTorch版本匹配
2. 需要CUDA支持以获得最佳性能
3. 首次运行时会自动下载和预处理数据
4. 生成的分子会保存为SDF格式


