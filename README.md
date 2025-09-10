# 3D Molecular Diffusion

基于扩散模型的3D分子生成项目。


## 环境要求

- Python = 3.9 / 3.10 / 3.11
- PyTorch 2.6.0
- CUDA 12.4

## 1. 环境配置(以python3.9为例)

### 创建python3.9虚拟环境
```
conda create -n py39 python=3.9
```
### 启动虚拟环境
```
conda activate py39
```


### 安装PyTorch和基础依赖
```
pip install torch==2.6.0  --index-url https://download.pytorch.org/whl/cu124 
```

```
pip install -r requirements.txt
```

### 安装PyTorch Geometric相关包
#### 注意：请确保PyTorch Geometric的版本与您的PyTorch和CUDA版本匹配。如果不匹配，请修改上述命令中的版本号。此处使用`cuda 12.4`对应在AutoDL的RTX4090中可选配置
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 生成分子

```
python scripts/sample_drug3d.py 
```


## 配置说明

- `configs/train_MolDiff_simple.yml`: 训练配置
- `configs/sample_MolDiff_simple.yml`: 采样配置

## 注意事项

1. 确保PyTorch Geometric版本与PyTorch版本匹配
2. 需要CUDA支持以获得最佳性能
3. 首次运行时会自动下载和预处理数据
4. 生成的分子会保存为SDF格式


