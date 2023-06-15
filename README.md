# Preemo LLMs in Prod Conference: Part II
Workshop on LLMs in Prod

## System Requirements
- CUDA Toolkit 117
- C++ Compiler (i.e. gcc 11.3.0)
- Python >= 3.10

## Getting Started
To install the python dependencies simply run
```bash
poetry install
```

In order to pull the `EleutherAI/lm-evaluation-harness` and `stanford-crfm/helm` you will need to run the following line:
```bash
git submodule update --init --recursive
```

## Downloading Models from HuggingFace
Although you can download models using `model.from_pretrained`, it's recommended to use the `git-lfs` to download the models.

```bash
git-lfs install
git clone https://huggingface.co/google/flan-ul2
```

## Fine-tuning on flan-ul2
To launch a fine-tuning job using _Deepspeed ZeRO3_ (without CPU offloading) run the below command:
```bash
accelerate launch --config_file ./finetune/launcher_configs/accelerate_zero3_no_offload_config.yaml ./finetune/zero3_gsm8k.py <model-path>
```

## Running Evals:
In order to conduct evaluations on the _GSM8K_ dataset run the below script.
`python lm-evaluation-harness/main.py --model hf-seq2seq --model_args pretrained=google/flan-ul2,dtype=float16,use_accelerate=True --task=gsm8k --batch_size=16 --write_out --output_base_path=./eval_results/ `

## Acknowledgement
The notebooks and code are inspired from various work from Hugging Face, EleutherAI, and stanford-crfm.