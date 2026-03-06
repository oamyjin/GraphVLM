## 🔥 Updates

- Added implementation for **GraphVLM (CVPR 2026)**.
- Initial release for **Graph-MLLM (arXiv)**.


------
<p align="center">
  <a href="#GraphVLM">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#Evaluation">Evaluation</a> •
  <a href="#Reference">Reference</a>
</p>



## Overview

Official implementation of **GraphVLM (CVPR 2026)** and the earlier **Graph-MLLM (arXiv)** version.

GraphVLM is a comprehensive benchmark for multimodal graph learning that extends state-of-the-art graph methods into the multimodal domain using VLMs. Fusing multimodal data with graph-structured data shows great promise for numerous real-world applications—such as social networks, healthcare, and recommendation systems—when nodes contain both textual and visual attributes.



## Installation

The implementation codes can be installed by running:
``` bash
git clone https://github.com/oamyjin/GraphVLM.git
```

## Evaluation
Our benchmark provides a fair, systematic comparison across three categories of baseline methods.
We propose three distinct strategies for integrating VLMs into these baselines. 
Collectively, these efforts enable existing baselines to effectively address multimodal graph learning tasks.

<hr style="border: 0.5px solid #ccc;" />

### VLM-as-Encoder

#### Step 1: Setup Environment

``` bash
conda create -n enhancer python=3.10
conda activate enhancer
cd Enhancer
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/th21_cu118 dgl
pip install pandas numpy scikit-learn
pip install -r requirement.txt
```

#### Step 2:  Data Preparation

Download our datasets from [huggingface](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/enhancer-dataset). And move the processed data to `./dataset`
```
.
├── configs
├── dataset
│   ├── Arts
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
│   ├── CD
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
│   └── pubmed
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
│   ├── cora
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
├── main.py
├── models.py
├── nc_dataset.py
├── requirements.txt
├── run_all.sh
```

#### Step 3: Run Evaluation
You can use the command to run Enhancer on datasets:
``` bash
./Enhancer/run_all.sh
``` 

<hr style="border: 0.5px solid #ccc;" />

### VLM-as-Aligner

#### Step 1: Setup Environment
<!-- We apply the VLM [Qwen-VL[(https://github.com/QwenLM/Qwen-VL) as the modality augmenter to convert visual information into text. 
Please refer to [here]() for Qwen-VL installation.   -->
Since this experiment involves two distinct baseline models, please follow their respective installation guides:
- [LLaGA](https://github.com/oamyjin/GraphVLM/tree/main/Augmenter/LLaGA): in `GraphVLM/Augmenter/LLaGA`
- [GraphPrompter](https://github.com/oamyjin/GraphVLM/tree/main/Augmenter/GraphPrompter#environment-setup): in `GraphVLM/Augmenter/GraphPrompter`

#### Step 2: Base Model Preparation
Please download the following base models for baselines:

- LLaGA: Please download [vicuna-7b-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k) and put it in `Augmenter/LLaGA/base_model`.
- GraphPrompter: Please download [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) and put it in 'Augmenter/GraphPrompter'.

#### Step 2: Data Preparation
The augmentation process follows the prompt mentioned in our benchmark paper.
Here, we prepare the augmented dataset in [huggingface](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/augmenter-dataset). Please download the processed dataset for evaluation and place the files into their corresponding dataset folders.

- [LLaGA](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/augmenter-dataset/llaga): update dataset to folder `datasets` under `GraphVLM/Augmenter/LLaGA`
- [GraphPrompter](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/augmenter-dataset/graphprompter): update dataset to folder `datasets` under `GraphVLM/Augmenter/GraphPrompter`


#### Step 3: Run Evaluation
Use the following command to run the augmented baselines for training, testing, and obtaining the final task accuracy:
``` bash
./Augmenter/LLaGA/train.sh
```
``` bash
./Augmenter/GraphPrompter/train.sh
``` 


<hr style="border: 0.5px solid #ccc;" />

### VLM-as-Predictor
#### Step 1: Setup Environment
<!-- We apply the VLM [Qwen-VL[(https://github.com/QwenLM/Qwen-VL) as the modality augmenter to convert visual information into text. 
Please refer to [here]() for Qwen-VL installation.   -->
Since this experiment involves two distinct baseline models, please follow their respective installation guides:
- [QWen-VL](https://github.com/oamyjin/GraphVLM/tree/main/Predictor/Qwen-VL): in `GraphVLM/Predictor/Qwen-VL`
- [LLaVA](https://github.com/oamyjin/GraphVLM/tree/main/Predictor/LLaVA): in `GraphVLM/Predictor/LLaVA`

#### Step 2: Base Model Preparation
Please download the following base models for baselines:

- QWen-VL: We utilize the QWen-VL-Chat version as the predictor and its model can be found from [here](https://huggingface.co/Qwen/Qwen-VL-Chat). Please update download it and put it in the path: `QWen-VL/local_model`.
- LLaVA: We also apply the LLaVA-v1.5-7B version as another predictor and its model can be downloaded from [here](https://huggingface.co/liuhaotian/llava-v1.5-7b). Please update it in the path:`LLaVA/local_model`.

#### Step 3: Data Preparation
Here, we prepare the prompt files for training and testing datasets on [huggingface](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/predictor-dataset). 
Please download the processed dataset for evaluation and place the files into their corresponding dataset folders.

- LLaVA: The testing dataset can be found [here](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/predictor-dataset/llava). Please download it and put it in `.Predictor/LLaVA/dataset`.
- QWen-VL: The training dataset can be found [here](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/predictor-dataset/qwen-vl/fintune_dataset).
  Please download it and put it in `./Predictor/Qwen-VL/finetune/fintune_dataset` for fine-tuning.
  Please also download the evaluation [prompt files](https://huggingface.co/datasets/oamyjin/GraphVLM/tree/main/predictor-dataset/qwen-vl/eval_prompt) in `./Predictor/Qwen-VL/eval_mm/mme/eval_prompt_files`.
  For other data in the testing dataset, QWen-VL uses the same as LLaVA, but with a different system prompt. 


#### Step 4: Run Evaluation
For LLaVA, we perform zero-shot testing using the codebase without any fine-tuning. Here is the command for evaluation:

``` bash
./Predictor/LLaVA/graphmllm_scripts/train_eval_movies.sh
```

For QWen-VL, we follow the official LoRA [training](https://github.com/QwenLM/Qwen-VL?tab=readme-ov-file#lora) finetuning instruction for supervised zero-shot and few-shot finetuning.

For a **quick start**, you can directly run the following commands for fine-tuning and evaluation:

- QWen-VL:
  
  ``` bash
  ./Predictor/Qwen-VL/finetune/my_finetune_lora_single_gpu_nb_image_tiltle.sh
  ./Predictor/Qwen-VL/eval_mm/mme/qwen_chat_eval.sh
  ```


## Reference
Our codes are implemented based on:

| **ID** | **Paper** | **Method** | **Thrust** | **Conference or Source** | **Github** |
|--------|---------|:----------:|:--------------:|:--------------:|:--------------:|
| 1      | [Multimodal Graph Benchmark](https://arxiv.org/pdf/1609.02907.pdf%EF%BC%89)     |    MLP, GCN, GraphSAGE     | Alignment Enhancer  | Arxiv  |   [github](https://github.com/mm-graph-benchmark/mm-graph-benchmark)    |
| 2      | [LLaGA: large language and graph assistant](https://arxiv.org/pdf/2402.08170) |    LLaGA     | VLM-as-Encoder |  ICML 2024    |   [github](https://github.com/mm-graph-benchmark/mm-graph-benchmark)    |
| 3      | [Can we Soft Prompt LLMs for Graph Learning Tasks?](https://arxiv.org/pdf/2402.10359) |    GraphPrompter     | VLM-as-Aligner |  WWW 2024    |[github](https://github.com/mm-graph-benchmark/mm-graph-benchmark)    |
| 4      | [LLaVA: Large Language and Vision Assistant](https://arxiv.org/pdf/2304.08485) |    LLaVA     | VLM-as-Predictor |  NeurIPS 2023    | [github](https://github.com/haotian-liu/LLaVA)    |
| 5      | [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/pdf/2308.12966) |    QWen-VL     | VLM-as-Predictor |  Arxiv    | [github](https://github.com/haotian-liu/LLaVA)    |
