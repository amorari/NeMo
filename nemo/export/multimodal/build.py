# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from time import time

import tensorrt as trt
import torch
import torch.nn as nn
import yaml
from tensorrt_llm.builder import Builder
from transformers import AutoModel
from omegaconf.omegaconf import OmegaConf

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model
from nemo.collections.multimodal.speech_llm.modules.perception_modules import AudioPerceptionModule
from nemo.core.classes.common import typecheck

logger = trt.Logger(trt.Logger.INFO)

def build_trtllm_engine(
    model_dir: str,
    visual_checkpoint_path: str,
    llm_checkpoint_path: str = None,
    model_type: str = "neva",
    llm_model_type: str = "llama",
    tensor_parallelism_size: int = 1,
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 1,
    max_multimodal_len: int = 1024,
    dtype: str = "bfloat16",
):
    trt_llm_exporter = TensorRTLLM(model_dir=model_dir, load_model=False)
    visual_checkpoint_model = ['neva', 'lita', 'vila', 'vita']
    trt_llm_exporter.export(
        nemo_checkpoint_path=visual_checkpoint_path if model_type in visual_checkpoint_model else llm_checkpoint_path,
        model_type=llm_model_type,
        tensor_parallelism_size=tensor_parallelism_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_prompt_embedding_table_size=max_multimodal_len,
        dtype=dtype,
        load_model=False,
    )

def build_salm_decoder_engine(
    engine_dir: str,
    nemo_checkpoint_path: str,
    llm_model_type: str = "llama",
    tensor_parallelism_size: int = 1,
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 1,
    max_multimodal_len: int = 1024,
    dtype: str = "bfloat16",
):
    trt_llm_exporter = TensorRTLLM(model_dir=engine_dir, load_model=False)
    trt_llm_exporter.export(
        nemo_checkpoint_path= nemo_checkpoint_path,
        model_type=llm_model_type,
        tensor_parallelism_size=tensor_parallelism_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_prompt_embedding_table_size=max_multimodal_len,
        dtype=dtype,
        load_model=False,
    )

def export_visual_wrapper_onnx(
    visual_wrapper, input, output_dir, input_names=['input'], dynamic_axes={'input': {0: 'batch'}}
):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.makedirs(f'{output_dir}/onnx', exist_ok=True)
    torch.onnx.export(
        visual_wrapper,
        input,
        f'{output_dir}/onnx/visual_encoder.onnx',
        opset_version=17,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )


def build_trt_engine(
    model_type,
    input_sizes,
    output_dir,
    vision_max_batch_size,
    dtype=torch.bfloat16,
    image_size=None,
    num_frames=None,
    nemo_config=None,
):
    part_name = 'visual_encoder'
    onnx_file = '%s/onnx/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s.engine' % (output_dir, part_name)
    config_file = '%s/%s' % (output_dir, "config.json")
    nemo_config_file = '%s/%s' % (output_dir, "nemo_config.yaml")

    with open(nemo_config_file, 'w') as f:
        yaml.dump(nemo_config, f)

    logger.log(trt.Logger.INFO, "Building TRT engine for %s" % part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {"precision": str(dtype).split('.')[-1], "model_type": model_type}
    if image_size is not None:
        config_args["image_size"] = image_size
    if num_frames is not None:
        config_args["num_frames"] = num_frames

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    # Delete onnx files since we don't need them now
    shutil.rmtree(f'{output_dir}/onnx')

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(vision_max_batch_size / 2))
    nMaxBS = vision_max_batch_size

    inputT = network.get_input(0)

    # input sizes can be a list of ints (e.g., [3, H, W]) when inputs are images,
    # or a list of three int lists (e.g., [[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]]).
    assert isinstance(input_sizes, list), "input_sizes must be a list"
    if isinstance(input_sizes[0], int):
        logger.log(trt.Logger.INFO, f"Processed input sizes {input_sizes}")
        inputT.shape = [nBS, *input_sizes]
        min_size = opt_size = max_size = input_sizes
    elif len(input_sizes) == 3 and isinstance(input_sizes[0], list):
        min_size, opt_size, max_size = input_sizes
        logger.log(trt.Logger.INFO, f"Processed min/opt/max input sizes {min_size}/{opt_size}/{max_size}")
    else:
        raise ValueError(f"invalid input sizes: {input_sizes}")

    profile.set_shape(inputT.name, [nMinBS, *min_size], [nOptBS, *opt_size], [nMaxBS, *max_size])
    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO, "Succeeded building %s in %d s" % (engine_file, t1 - t0))
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

    Builder.save_config(config_wrapper, config_file)


def build_neva_engine(
    model_type: str,
    model_dir: str,
    visual_checkpoint_path: str,
    vision_max_batch_size: int = 1,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # extract NeMo checkpoint
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        mp0_weights, nemo_config, _ = load_nemo_model(visual_checkpoint_path, temp_path)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class DownSampleBlock(torch.nn.Module):
        def forward(self, x):
            vit_embeds = x
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.flat_square(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            return vit_embeds

        def flat_square(self, x):
            n, w, h, c = x.size()
            if w % 2 == 1:
                x = torch.cat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
                n, w, h, c = x.size()
            if h % 2 == 1:
                x = torch.cat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
                n, w, h, c = x.size()
            x = x.view(n, w, int(h / 2), int(c * 2))
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
            return x

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            vision_x = self.encoder(pixel_values=images, output_hidden_states=True)
            vision_x = vision_x.hidden_states[-2]
            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(
        vision_config["from_pretrained"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    if nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "mlp2x_gelu":
        vision_connector = torch.nn.Sequential(
            torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(nemo_config["hidden_size"], nemo_config["hidden_size"], bias=True),
        ).to(dtype=dtype)

        key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
        for layer in range(0, 3, 2):
            vision_connector[layer].load_state_dict(
                {
                    'weight': mp0_weights[f"{key_prefix}.{layer}.weight"].to(dtype),
                    'bias': mp0_weights[f"{key_prefix}.{layer}.bias"].to(dtype),
                }
            )
    elif nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "linear":
        vision_connector = torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True)
        key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
        vision_connector.load_state_dict(
            {
                'weight': mp0_weights[f"{key_prefix}.weight"].to(dtype),
                'bias': mp0_weights[f"{key_prefix}.bias"].to(dtype),
            }
        )
    elif nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "mlp_downsample":
        vision_connector = torch.nn.Sequential(
            DownSampleBlock(),
            torch.nn.LayerNorm(vision_config["hidden_size"] * 4),
            torch.nn.Linear(vision_config["hidden_size"] * 4, nemo_config["hidden_size"], bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(nemo_config["hidden_size"], nemo_config["hidden_size"], bias=True),
        ).to(dtype=dtype)
        key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
        for layer in [1, 2, 4]:
            vision_connector[layer].load_state_dict(
                {
                    'weight': mp0_weights[f"{key_prefix}.{layer}.weight"].to(dtype),
                    'bias': mp0_weights[f"{key_prefix}.{layer}.bias"].to(dtype),
                }
            )

    else:
        raise ValueError(f"Unknown projector type: {nemo_config['mm_cfg']['mm_mlp_adapter_type']}")

    # export the whole wrapper
    lita_num_frames = None
    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector).to(device, dtype)
    if model_type == "lita" or model_type == "vila":
        image_size = hf_config.image_size
        if model_type == "lita":
            lita_num_frames = nemo_config['mm_cfg']['lita']['sample_frames']
    else:
        image_size = hf_config.vision_config.image_size
        if model_type == "vita":
            lita_num_frames = nemo_config['mm_cfg']['lita']['sample_frames']
    dummy_image = torch.empty(
        1, 3, image_size, image_size, dtype=dtype, device=device
    )  # dummy image shape [B, C, H, W]

    export_visual_wrapper_onnx(wrapper, dummy_image, model_dir)
    build_trt_engine(
        model_type,
        [3, image_size, image_size],
        model_dir,
        vision_max_batch_size,
        dtype,
        image_size=image_size,
        num_frames=lita_num_frames if model_type == "lita" or model_type == 'vita' else None,
        nemo_config=nemo_config,
    )


def build_video_neva_engine(
    model_dir: str,
    visual_checkpoint_path: str,
    vision_max_batch_size: int = 1,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # extract NeMo checkpoint
    with tarfile.open(visual_checkpoint_path) as tar:
        nemo_config = yaml.safe_load(tar.extractfile("./model_config.yaml"))
        try:
            # trained without TP
            mp0_weights = torch.load(tar.extractfile("./model_weights.ckpt"), map_location=device)
        except KeyError:
            # trained with TP
            mp0_weights = torch.load(tar.extractfile("./mp_rank_00/model_weights.ckpt"), map_location=device)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            b, num_frames, c, h, w = images.shape
            images = images.view(b * num_frames, c, h, w)
            vision_x = self.encoder(pixel_values=images, output_hidden_states=True)  # [(B num_frames), C, H, W]
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]

            # reshape back to [B, num_frames, img_size, hidden_size]
            vision_x = vision_x.view(b, num_frames, -1, vision_x.shape[-1])

            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(
        vision_config["from_pretrained"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    assert nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "linear"
    vision_connector = torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True)

    key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
    vision_connector.load_state_dict(
        {
            'weight': mp0_weights[f"{key_prefix}.weight"].to(dtype),
            'bias': mp0_weights[f"{key_prefix}.bias"].to(dtype),
        }
    )

    # export the whole wrapper
    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector).to(device, dtype)
    image_size = hf_config.vision_config.image_size
    num_frames = nemo_config['data']['num_frames']
    dummy_video = torch.empty(1, num_frames, 3, image_size, image_size, dtype=dtype, device=device)  # dummy image
    export_visual_wrapper_onnx(wrapper, dummy_video, model_dir)
    build_trt_engine(
        "video-neva",
        [num_frames, 3, image_size, image_size],  # [num_frames, 3, H, W]
        model_dir,
        vision_max_batch_size,
        dtype,
        image_size=image_size,
        num_frames=num_frames,
    )

def build_multimodal_engine(
    model_dir: str,
    checkpoint_path: str,
    model_type: str = "neva",
    max_batch_size: int = 1,
):
    model_list = ['neva', 'lita', 'vila', 'vita', 'salm']
    if model_type in model_list:
        build_neva_engine(model_type, model_dir, checkpoint_path, max_batch_size)
    elif model_type == "video-neva":
        build_video_neva_engine(model_dir, checkpoint_path, max_batch_size)
    elif model_type == "salm":
        build_salm_engine(model_dir, checkpoint_path, max_batch_size, dtype='bfloat16')
    else:
        raise RuntimeError(f"Invalid model type {model_type}")



def build_salm_engine(
    model_dir: str,
    nemo_dir: str,
    max_batch_size: int,
    dtype: str = "bfloat16",
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    #Load main nemo config
    with open(f'{nemo_dir}/config.yaml') as f:
        nemo_config = OmegaConf.load
    perception_config = nemo_config['perception']

    # Load the perception model
    perception_state_dict = load_state_dict_from_nemo(
        nemo_dir= nemo_dir,
        map_location= device)

    #TODO: Using Nemo code here, need to rewrite the preprocessor to be exportable to TensorRT
    perception = AudioPerceptionModule(cfg=perception_config)
    perception.load_state_dict(perception_state_dict)
    # verify if the exported perception model is correct
    perception.eval()
    print(perception(input_signal = torch.randn(1, 1000), input_signal_length = torch.tensor([1000])))

    class EncodersModuleWrapper(torch.nn.Module):
        def __init__(self, encoder, modality_adapter, projector):
            super().__init__()
            self.encoder = encoder
            self.modality_adapter = modality_adapter
            self.proj = projector

        @typecheck.disable_checks()
        def forward(
            self,
            processed_signal=None,
            processed_signal_length=None,
        ):
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
            encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)

            # b, c, t -> b, t, c
            encoded = self.proj(encoded.transpose(1, 2))
            return encoded, encoded_len

    if 'output_dim' not in perception_config.modality_adapter and "d_model" in perception_config.modality_adapter:  # e.g., conformer encoder
        proj = nn.Linear(perception_config.modality_adapter.d_model, perception_config.output_dim)
    else:
        proj = nn.Identity()         

    encodersModuleWrapper = EncodersModuleWrapper(
        encoder = perception.encoder, 
        modality_adapter = perception.modality_adapter, 
        projector= proj)

    #Test with dummy input
    input_features = perception_config.encoder.feat_in
    processed_signal = torch.randn(1, input_features, 128)
    processed_signal_length = torch.tensor([128])
    print(f"Encoders Wrapper input features: {input_features}")
    print(f"Encoders Wrapper processed signal shape: {processed_signal.shape}")
    print(f"Encoders Wrapper processed signal length : {processed_signal_length}")
    print(encodersModuleWrapper(processed_signal, processed_signal_length))

    onnx_file = f'{model_dir}/encoder/model.onnx'
    export_wrapper_to_onnx(
        wrapper= encodersModuleWrapper, 
        input= (processed_signal, processed_signal_length),
        filepath= onnx_file,
        input_names=['processed_signal', 'processed_signal_length'],
        output_names=['encoded', 'encoded_len'],
        dynamic_axes= {
                'processed_signal': {0: 'batch_size', 2: 'time_frames'},
                'processed_signal_length': {0: 'batch_size'},
            }
        )
    
    build_salm_encoder_engine(
        nemo_config= perception_config,
        engine_dir = f'{model_dir}/encoder',
        onnx_file= onnx_file,
        dtype= dtype,
        max_batch_size= max_batch_size,
    )

    #eliminate onnx file since we don't need it anymore
    os.remove(onnx_file)

   
    #TODO: read these values from a config file
    build_salm_decoder_engine(
        engine_dir=f'{model_dir}/decoder',
        nemo_checkpoint_path=f'{nemo_dir}/decoder.nemo',
        llm_model_type = "llama",
        tensor_parallelism_size = nemo_config['tensor_model_parallel_size'],
        max_input_len = 512,
        max_output_len = 256,
        max_batch_size = max_batch_size,
        max_multimodal_len = 512,
        dtype = dtype,
    )



from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils.model_utils import inject_model_parallel_rank
import argparse

def load_state_dict_from_nemo(nemo_dir, map_location):
    save_restore_connector = NLPSaveRestoreConnector()

    model_weights = f"{nemo_dir}/model_weights.ckpt"
    model_weights = inject_model_parallel_rank(model_weights)
    state_dict = save_restore_connector._load_state_dict_from_disk(
        model_weights, map_location=map_location
    )
    return state_dict


def get_config_and_state_dict_from_nemo(filepath, map_location, output_dir, sharded_state_dict=None):
        cwd = os.getcwd()
        save_restore_connector = NLPSaveRestoreConnector()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                if os.path.isfile(filepath):
                    save_restore_connector._unpack_nemo_file(path2file=filepath, out_folder=tmpdir)
                else:
                    tmpdir = filepath

                os.chdir(tmpdir)
                config_yaml = "model_config.yaml"
                model_weights_ckpt = "model_weights.ckpt"
                
                # find file in tmpdir that endswith "tokenizer.model"
                for file in os.listdir(tmpdir):
                    if file.endswith("tokenizer.model"):
                        tokenizer = file
                        break
                tokenizer_path = os.path.join(tmpdir, tokenizer)
                # copy tokenizer_path to current directory
                os.system(f"cp {tokenizer_path} {output_dir}")
                tokenizer_path = os.path.join(output_dir, tokenizer)

                # load conf
                with open(config_yaml) as f:
                    conf = OmegaConf.load(f)
                
                os.chdir(cwd)
                model_weights = os.path.join(tmpdir, model_weights_ckpt)
                model_weights = inject_model_parallel_rank(model_weights)
                state_dict = save_restore_connector._load_state_dict_from_disk(
                    model_weights, map_location=map_location
                )

                # distributed checkpointing
                if state_dict is None and sharded_state_dict is not None:
                    checkpoint = dict(state_dict=sharded_state_dict)
                    tmp_model_weights_ckpt = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                    tmp_model_weights_dir = os.path.splitext(tmp_model_weights_ckpt)[0]
                    assert os.path.isdir(tmp_model_weights_dir), f'Expected {tmp_model_weights_dir} to be a directory.'
                    checkpoint = dist_checkpointing.load(
                        sharded_state_dict=checkpoint,
                        checkpoint_dir=tmp_model_weights_dir,
                    )
                    state_dict = checkpoint["state_dict"]

                conf.tokenizer.model = tokenizer_path
                return conf, state_dict
            finally:
                os.chdir(cwd)



def get_perception_state_dict(state_dict):
    perception_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("perception."):
            key = key.replace("perception.", "", 1)
            perception_state_dict[key] = value
    return perception_state_dict

def build_salm_encoder_engine(
    nemo_config,
    engine_dir,
    onnx_file,
    dtype,
    max_batch_size,
):

    output_config_file = f'{engine_dir}/config.json',
    output_engine_file = f'{engine_dir}/rank0.trt',
    logger.log(trt.Logger.INFO, "Building TRT encoder engine ")

   # Other relevant parameters
    #max_workspace_size = 1 << 30  # 1GB, adjust as needed

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    config_args = nemo_config
    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config

    if dtype == trt.DataType.HALF:
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == trt.DataType.BF16:
        config.set_flag(trt.BuilderFlag.BF16)

    # Set other flags based on your YAML configuration
    #if nemo_config['model'].get('tensor_model_parallel_size', 1) > 1:
    #    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    #if nemo_config['trainer'].get('gradient_clip_val', 0.0) > 0:
    #    nemo_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        else:
            logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    profile = builder.create_optimization_profile()

    min_batch_size = 1
    opt_batch_size = max(min_batch_size,int(max_batch_size/2))
    input_features = nemo_config['perception']['encoder']['feat_in']
    max_length = nemo_config['encoder_seq_length']

    # Processed_input shape
    min_processed_input_dims = (min_batch_size, input_features, 1)
    opt_processed_input_dims = (opt_batch_size, input_features, max(1,int(max_length/2)))
    max_processed_input_dims = (max_batch_size, input_features, max_length)
    print(f"Processed input shape optimization: {min_processed_input_dims}/{opt_processed_input_dims}/{max_processed_input_dims}")
    input_tensor_0 = network.get_input(0)
    profile.set_shape(input_tensor_0.name, 
                min_processed_input_dims, 
                opt_processed_input_dims,
                max_processed_input_dims)

    # Processed_input_length shape
    input_tensor_1 = network.get_input(1)
    profile.set_shape(input_tensor_1.name, 
                      trt.Dims([min_batch_size]),
                      trt.Dims([opt_batch_size]), 
                      trt.Dims([max_batch_size]))

    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (output_engine_file))
    else:
        logger.log(trt.Logger.INFO, "Succeeded building %s in %d s" % (output_engine_file, t1 - t0))
        with open(output_engine_file, 'wb') as f:
            f.write(engine_string)

    Builder.save_config(config_wrapper, output_config_file)

def export_wrapper_to_onnx(
    wrapper, input, filepath, input_names=['input'], output_names=['output'], dynamic_axes=None
):
    logger.log(trt.Logger.INFO, f"Exporting to ONNX format and saving in {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.onnx.export(
        model= wrapper,
        args= input,
        f=filepath,
        input_names= input_names,
        output_names= output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes or {},
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build multimodal engine")
    parser.add_argument("--model_dir", type=str, help="Path to the model directory, where the engines will be saved")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the nemo checkpoint files")
    parser.add_argument("--model_type", type=str, default="neva", help="Model type (default: neva)")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size (default: 1)")

    args = parser.parse_args()

    build_multimodal_engine(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        max_batch_size=args.max_batch_size,
    )