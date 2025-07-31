# 使用ChatGLM3-6B进行言行一致性测试

本仓库提供了使用ChatGLM3-6B模型通过LoRA微调重现论文 *“Large Language Models Often Say One Thing and Do Another”* 实验的说明和代码。项目通过Word-Deed Consistency Test (WDCT)数据集评估模型的言语（words）与行为（deeds）一致性。

## 目录
- [前置要求](#前置要求)
- [环境配置](#环境配置)
- [模型与代码设置](#模型与代码设置)
- [运行言行一致性测试](#运行言行一致性测试)
  - [基础模型评估](#基础模型评估)
  - [SFT监督微调（言语对齐）](#sft监督微调言语对齐)
  - [DPO直接偏好优化](#dpo直接偏好优化)
  - [微调模型评估](#微调模型评估)
- [补充实验](#补充实验)
  - [行为对齐微调](#行为对齐微调)
  - [思维链提示](#思维链提示)
  - [温度与多轮测试](#温度与多轮测试)
- [结果](#结果)
- [使用工具](#使用工具)

## 前置要求
- **Python**：版本3.10
- **CUDA**：版本12.1
- **硬件**：支持LoRA微调的GPU
- **软件**：Conda、Git、VSCode、Excel、[TablesGenerator](https://www.tablesgenerator.com/)、Overleaf

## 环境配置
1. **创建并激活Conda环境**：
   ```bash
   conda create -y -n wdct-test python=3.10
   conda activate wdct-test
   ```

2. **克隆仓库**：
   ```bash
   git clone https://github.com/icip-cas/Word-Deed-Consistency-Test.git
   cd Word-Deed-Consistency-Test
   ```

3. **安装依赖**：
   LoRA微调所需依赖如下：
   - fastchat==0.1.0
   - ipykernel==6.23.1
   - numpy==1.24.3
   - tokenizers==0.19.1
   - torch==2.4.0
   - tqdm==4.65.0
   - transformers==4.44.2
   - datasets==2.12.0
   - beautifulsoup4==4.12.2
   - wandb==0.15.3
   - hydra-core==1.3.2
   - tensor-parallel==1.2.4
   - scipy>=1.11.3
   - peft==0.12.0

   安装命令：
   ```bash
   pip install -r requirements.txt
   ```

4. **下载ChatGLM3-6B模型**：
   进入模型目录，从ModelScope克隆模型：
   ```bash
   cd /home/blcuzfy2024/Word-Deed-Consistency-Test/dpo/config/model
   git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
   ```

## 模型与代码设置
1. **配置模型**：
   - 在`Word-Deed-Consistency-Test/eval/test_local_models.py`中更新模型名称：
     ```python
     parser.add_argument('--model', default='chatglm3-6b', type=str, help='Model to be evaluated.')
     ```
   - 在`Word-Deed-Consistency-Test/utils/config.py`中指定模型路径：
     ```python
     'chatglm3-6b': '/home/blcuzfy2024/Word-Deed-Consistency-Test/dpo/config/model/chatglm3-6b'
     ```

## 运行言行一致性测试
### 基础模型评估
#### 方法一：
1. 进入项目目录：
   ```bash
   cd /home/blcuzfy2024/Word-Deed-Consistency-Test
   ```
2. 运行评估：
   ```bash
   python eval/test.py
   ```
3. 结果保存至：
   - `/home/blcuzfy2024/Word-Deed-Consistency-Test/result/metrics_normal.csv`
   - `/home/blcuzfy2024/Word-Deed-Consistency-Test/result/WDCT_result_chatglm3-6b_0_normal.csv`

#### 方法二：
直接在终端运行：
```bash
CUDA_VISIBLE_DEVICES=0 python eval/test_local_models.py \
    --model chatglm3-6b \
    --dataset WDCT \
    --batch_size 16 \
    --temperature 0 \
    --max_new_tokens 10 \
    --run_time 1
```
结果保存在`result/`目录，包括：
- `metrics_chatglm3-6b_normal.csv`：总体评估指标（如平均一致性得分、PCS）。
- `WDCT_NE_result_chatglm3-6b_0.0_normal.csv`：逐条评估结果。

### SFT监督微调（言语对齐）
运行监督微调（SFT）：
```bash
python train.py \
    model=chatglm3-6b \
    exp_name=chatglm3-6b-speak-sft \
    datasets=[WDCT] \
    mix_dataset=alpaca \
    mix_ratio=9 \
    change_mode=speak \
    loss=dpo \
    lr=5e-7 \
    n_epochs=4
```

### DPO直接偏好优化
在SFT模型基础上运行DPO：
```bash
python train.py \
    model=chatglm3-6b \
    exp_name=chatglm3-6b-speak-dpo \
    model.archive=/home/blcuzfy2024/Word-Deed-Consistency-Test/output/chatglm3-6b-speak-sft/LATEST \
    datasets=[WDCT_NE] \
    change_mode=speak \
    loss=dpo \
    lr=5e-7 \
    n_epochs=4
```
输出权重保存至：
- `output/chatglm3-6b-lora-dpo/LATEST/`
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `tokenization_chatglm.py`
  - `tokenizer.model`
  - `tokenizer_config.json`

### 微调模型评估
评估第4轮SFT和DPO模型：
```bash
python eval/test_local_models.py \
    --model chatglm3-6b \
    --dataset WDCT_NE \
    --ckpt_path "/home/blcuzfy2024/Word-Deed-Consistency-Test/output/chatglm3-6b-speak-sft/LATEST" \
    --batch_size 16 \
    --temperature 0 \
    --max_new_tokens 10 \
    --run_time 1
```
基础模型使用更大批次大小：
```bash
python eval/test_local_models.py \
    --model chatglm3-6b-base \
    --dataset WDCT_NE \
    --batch_size 32 \
    --temperature 0 \
    --max_new_tokens 10 \
    --run_time 5
```
结果保存方式同基础模型评估。

## 补充实验
### 行为对齐微调
将`change_mode`改为`act`进行微调：
```bash
python train.py \
    model=chatglm3-6b \
    exp_name=chatglm3-6b-act-sft \
    datasets=[WDCT] \
    mix_dataset=alpaca \
    mix_ratio=9 \
    change_mode=act \
    loss=dpo \
    lr=5e-7 \
    n_epochs=4
```

### 思维链提示
使用3-shot思维链（CoT）提示，调整`test_setting`并增加`max_new_tokens`：
```bash
python eval/test_local_models.py \
    --model chatglm3-6b \
    --dataset WDCT_NE \
    --test_setting 3_shot_cot \
    --batch_size 16 \
    --temperature 0 \
    --max_new_tokens 50 \
    --run_time 1
```

### 温度与多轮测试
测试不同温度（0、0.5、1）下5轮一致性：
```bash
python eval/test_local_models.py \
    --model chatglm3-6b \
    --dataset WDCT_NE \
    --batch_size 16 \
    --temperature 0 \
    --max_new_tokens 10 \
    --sample_num 100 \
    --run_time 5
```

## 结果
结果保存在`result/`目录，以CSV格式存储总体指标和逐条一致性得分。可使用Excel或[TablesGenerator](https://www.tablesgenerator.com/)将结果转为LaTeX格式，在Overleaf中生成报告。

## 使用工具
- **VSCode**：代码编辑与脚本执行
- **Excel**：数据分析与可视化
- **[TablesGenerator](https://www.tablesgenerator.com/)**：Excel表格转LaTeX
- **Overleaf**：LaTeX文档编辑与报告生成