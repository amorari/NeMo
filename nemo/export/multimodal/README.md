## Setting up the environment
This code needs to be executed with both Nemo and TRT-LLM support. This container can be used to execute it:

gitlab-master.nvidia.com:5005/dl/joc/nemo-ci/trtllm_0.11/train:pipe.16718524-x86

You can also rerun ./reinstall.sh in Nemo so that the nemo package is using the current Nemo folder.

## Separate Nemo components
In this part, we are going to export SALM model into TRTLLM.
First, let's download the [SALM nemo model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speechllm_fc_llama2_7b/) from NVIDIA ngc.

```
mkdir -p nemo_model 
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/speechllm_fc_llama2_7b/1.23.1/files?redirect=true&path=speechllm_fc_llama2_7b.nemo' -O nemo_model/speechllm_fc_llama2_7b.nemo
```

Then, we need to extract the different parts of SALM.
```
python3 separate_salm_weights.py --model_file_path=nemo_model/speechllm_fc_llama2_7b.nemo --output_dir=nemo_output
```
It takes a while to run the above command.

Under the `nemo_model` dir, you'll see:
```
nemo_model
    |___config.yaml
    |___lora.nemo
    |___perception
    |         |____model_config.yaml
    |         |____model_weights.ckpt
    |___llm.nemo
    |___ xxx.tokenizer.model
```

After we get the lora nemo model and llm nemo model, we can merge the lora part into the llm by:
```
python /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py \
    trainer.accelerator=gpu \
    tensor_model_parallel_size=1 \
    pipeline_model_parallel_size=1 \
    gpt_model_file=nemo_model/llm.nemo \
    lora_model_path=nemo_model/lora.nemo \
    merged_model_path=nemo_model/llm_merged.nemo
```

## Create TensorRT engines

```
python build.py --model_dir trt_engines --checkpoint_path nemo_model --model_type salm --max_batch_size 1
```

After we should obtain the engines as follows:

trt_engines
    |___encoder
    |       |____config.json
    |       |____rank0.trt
    |___decoder
            |____config.json
            |____rank0.trt


