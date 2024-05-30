# 安全数据集处理工具

## 准备工作

### 数据集
首先下载官方提供的数据集 `安全.csv`，并放置于 `data` 目录下。

### 环境
确保安装了以下Python库：
```bash
pip install transformers torch openai
```

## 使用方法

该工具的使用分为三个步骤：生成、测评和总结。

### 1. 生成
需要提供模型路径（本地路径或Hugging Face远程仓库）、数据文件路径和生成结构路径。

```bash
python code/generate.py $model_dir $data_file $save_dir
```

例如：
```bash
python code/generate.py lmsys/vicuna-7b-v1.5 data/安全.csv results
```

或者使用本地模型路径：
```bash
python code/generate.py ${local_model_dir}/vicuna-7b-v1.5 data/安全.csv results
```

### 2. 测评
基于GPT来对生成结果进行分类，需要在 `evaluate.py` 中提供 `openai.api_key`。

```bash
python code/evaluate.py $model_dir $data_file $save_dir
```

### 3. 总结
会以平均正确率的形式汇总各个类别的比例，其中完全遵从占比越高，性能越好；完全拒绝和部分拒绝之和占比越低，性能越好。

```bash
python code/conclusion.py $model_dir $data_file $save_dir
```

## 注意事项
- 确保在使用工具前已经正确安装了所有依赖。
- 根据需要配置好模型路径和API密钥。
- 根据实际情况调整命令行中的参数。

## 致谢
本项目测评方法是借鉴：XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models
其项目地址为：https://github.com/paul-rottger/exaggerated-safety