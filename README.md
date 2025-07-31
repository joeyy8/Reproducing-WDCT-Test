# Word-Deed Consistency Test with ChatGLM3-6B

This repository provides instructions and code to reproduce the experiments from the paper *"Large Language Models Often Say One Thing and Do Another"* using the ChatGLM3-6B model with LoRA fine-tuning. The project evaluates the consistency between a model's stated beliefs (words) and its actions (deeds) using the Word-Deed Consistency Test (WDCT) dataset.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Model and Code Setup](#model-and-code-setup)
- [Running the Word-Deed Consistency Test](#running-the-word-deed-consistency-test)
  - [Base Model Evaluation](#base-model-evaluation)
  - [SFT Fine-Tuning (Speak Alignment)](#sft-fine-tuning-speak-alignment)
  - [DPO Fine-Tuning](#dpo-fine-tuning)
  - [Fine-Tuned Model Evaluation](#fine-tuned-model-evaluation)
- [Additional Experiments](#additional-experiments)
  - [Act Alignment Fine-Tuning](#act-alignment-fine-tuning)
  - [Chain-of-Thought Prompting](#chain-of-thought-prompting)
  - [Temperature and Multi-Run Testing](#temperature-and-multi-run-testing)
- [Results](#results)
- [Tools Used](#tools-used)

## Prerequisites
- **Python**: Version 3.10
- **CUDA**: Version 12.1
- **Hardware**: GPU with sufficient memory for LoRA fine-tuning
- **Software**: Conda, Git, VSCode, Excel, [TablesGenerator](https://www.tablesgenerator.com/), Overleaf

## Environment Setup
1. **Create and activate Conda environment**:
   ```bash
   conda create -y -n wdct-test python=3.10
   conda activate wdct-test
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/icip-cas/Word-Deed-Consistency-Test.git
   cd Word-Deed-Consistency-Test
   ```

3. **Install dependencies**:
   The required dependencies for LoRA fine-tuning are:
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

   Install them using:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the ChatGLM3-6B model**:
   Navigate to the model directory and clone the model from ModelScope:
   ```bash
   cd /home/blcuzfy2024/Word-Deed-Consistency-Test/dpo/config/model
   git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
   ```

## Model and Code Setup
1. **Configure the model**:
   - Update the model name in `Word-Deed-Consistency-Test/eval/test_local_models.py`:
     ```python
     parser.add_argument('--model', default='chatglm3-6b', type=str, help='Model to be evaluated.')
     ```
   - Specify the model path in `Word-Deed-Consistency-Test/utils/config.py`:
     ```python
     'chatglm3-6b': '/home/blcuzfy2024/Word-Deed-Consistency-Test/dpo/config/model/chatglm3-6b'
     ```

## Running the Word-Deed Consistency Test
### Base Model Evaluation
#### Method 1:
1. Navigate to the project directory:
   ```bash
   cd /home/blcuzfy2024/Word-Deed-Consistency-Test
   ```
2. Run the evaluation:
   ```bash
   python eval/test.py
   ```
3. Results will be saved in:
   - `/home/blcuzfy2024/Word-Deed-Consistency-Test/result/metrics_normal.csv`
   - `/home/blcuzfy2024/Word-Deed-Consistency-Test/result/WDCT_result_chatglm3-6b_0_normal.csv`

#### Method 2:
Run directly from the terminal:
```bash
CUDA_VISIBLE_DEVICES=0 python eval/test_local_models.py \
    --model chatglm3-6b \
    --dataset WDCT \
    --batch_size 16 \
    --temperature 0 \
    --max_new_tokens 10 \
    --run_time 1
```
Results will be saved in the `result/` directory, including:
- `metrics_chatglm3-6b_normal.csv`: Overall evaluation metrics (e.g., average consistency score, PCS).
- `WDCT_NE_result_chatglm3-6b_0.0_normal.csv`: Per-sample evaluation results.

### SFT Fine-Tuning (Speak Alignment)
Run supervised fine-tuning (SFT) with the following command:
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

### DPO Fine-Tuning
Run direct preference optimization (DPO) on the SFT model:
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
Output weights will be saved in:
- `output/chatglm3-6b-lora-dpo/LATEST/`
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `tokenization_chatglm.py`
  - `tokenizer.model`
  - `tokenizer_config.json`

### Fine-Tuned Model Evaluation
Evaluate the fine-tuned models (4th epoch of SFT and DPO):
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
For the base model with larger batch size:
```bash
python eval/test_local_models.py \
    --model chatglm3-6b-base \
    --dataset WDCT_NE \
    --batch_size 32 \
    --temperature 0 \
    --max_new_tokens 10 \
    --run_time 5
```
Results are saved similarly to the base model evaluation.

## Additional Experiments
### Act Alignment Fine-Tuning
To fine-tune with `change_mode=act`:
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

### Chain-of-Thought Prompting
To test with 3-shot Chain-of-Thought (CoT) prompting, modify `test_setting` and increase `max_new_tokens`:
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

### Temperature and Multi-Run Testing
Test consistency across different temperatures (0, 0.5, 1) with 5 runs:
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

## Results
Results are stored in the `result/` directory as CSV files containing overall metrics and per-sample consistency scores. Use Excel or [TablesGenerator](https://www.tablesgenerator.com/) to convert results to LaTeX for reporting in Overleaf.

## Tools Used
- **VSCode**: Code editing and script execution
- **Excel**: Data analysis and visualization
- **[TablesGenerator](https://www.tablesgenerator.com/)**: Convert Excel tables to LaTeX
- **Overleaf**: LaTeX document preparation for reporting