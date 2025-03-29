#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : humeng
# @Time    : 2021/12/8
import multiprocessing
from os import environ
import onnxruntime
from psutil import cpu_count
from transformers import BertTokenizer, AlbertModel
import torch

environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))  # OMP 的线程数
environ["OMP_WAIT_POLICY"] = 'ACTIVE'  # 开启OpenMP加速运行。

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from transformers import BertTokenizerFast


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    # 在ONNX Runtime中通过设置特定的SessionOptions会自动使用大多数优化
    options = SessionOptions()
    # Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose.
    options.intra_op_num_threads = multiprocessing.cpu_count()
    # Graph optimization level for this session(图结构优化设置)
    # ORT_DISABLE_ALL -> 取消所有的 optimizations
    # ORT_ENABLE_BASIC -> 启用 basic optimizations
    # ORT_ENABLE_EXTENDED -> 启用 extended optimizations
    # ORT_ENABLE_ALL -> 启用所有优化设置
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()
    buffer.append(end - start)


@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str


def try_norm_qua():
    """
    测试量化版本和常规版本
    Returns:

    """
    model_ckpt = "clue/albert_chinese_small"
    torch_model = AlbertModel.from_pretrained(model_ckpt)
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    torch_model.eval()
    model_inputs = tokenizer("元宇宙虚拟地块 430 万美元，为虚拟房产的最高成交价，为什么虚拟地块能卖这么贵？谁在为虚拟地块买单？",
                             return_tensors="pt")
    # Warm up
    input_ids, attention_mask, token_type_ids = model_inputs.data['input_ids'], model_inputs.data['token_type_ids'], \
                                                model_inputs.data['attention_mask']
    torch_out = torch_model(input_ids, attention_mask, token_type_ids)

    t0 = time()
    for _ in range(50):
        torch_out = torch_model(input_ids, attention_mask, token_type_ids)
    print('normal runtime:', time() - t0)

    # Quantize量化 --
    torch_model = torch.quantization.quantize_dynamic(
        model=torch_model.to("cpu"), qconfig_spec={torch.nn.Bilinear}, dtype=torch.qint8
    )
    torch_model.eval()

    model_inputs = tokenizer("元宇宙虚拟地块 430 万美元，为虚拟房产的最高成交价，为什么虚拟地块能卖这么贵？谁在为虚拟地块买单？",
                             return_tensors="pt")
    # Warm up
    input_ids, attention_mask, token_type_ids = model_inputs.data['input_ids'], model_inputs.data['token_type_ids'], \
                                                model_inputs.data['attention_mask']
    torch_out = torch_model(input_ids, attention_mask, token_type_ids)

    t0 = time()
    for _ in range(50):
        torch_out = torch_model(input_ids, attention_mask, token_type_ids)
    print('quantized runtime:', time() - t0)


# 在导出模型之前必须调用 model.eval() 或 model.train(False)，因为这会将模型设置为“推理模式”。 这是必需的，因为 dropout 或 batchnorm 等运算符在推理和训练模式下的行为有所不同。
def gen_norm_onnx():
    """
    模型转onnx
    Returns:

    """
    model_ckpt = "clue/albert_chinese_small"
    torch_model = AlbertModel.from_pretrained(model_ckpt)
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    torch_model.eval()
    model_inputs = tokenizer("元宇宙虚拟地块 430 万美元，为虚拟房产的最高成交价，为什么虚拟地块能卖这么贵？谁在为虚拟地块买单？",
                             return_tensors="pt")
    # Warm up
    input_ids, attention_mask, token_type_ids = model_inputs.data['input_ids'], model_inputs.data['token_type_ids'], \
                                                model_inputs.data['attention_mask']
    torch_out = torch_model(input_ids, attention_mask, token_type_ids)

    t0 = time()
    for _ in range(50):
        torch_out = torch_model(input_ids, attention_mask, token_type_ids)
    print('normal runtime:', time() - t0)

    torch.onnx.export(torch_model,  # model being run
                      args=(input_ids, attention_mask, token_type_ids),  # model input (or a tuple for multiple inputs)
                      f="onnx/albert_chinese_small_qu.onnx",
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization, 是否执行常量折叠优化
                      input_names=['input_ids', 'token_type_ids', 'attention_mask'],  # the model's input names
                      output_names=['output'],  # the model's output names, 按顺序分配名称到图中的输出节点
                      dynamic_axes={'input_ids': {0: 'batch_size'},  # variable lenght axes
                                    'token_type_ids': {0: 'batch_size'},
                                    'attention_mask': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


def gen_qu_onnx():
    """
    模型转onnx
    Returns:

    """
    model_ckpt = "clue/albert_chinese_small"
    torch_model = AlbertModel.from_pretrained(model_ckpt)
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    # Quantize
    # 模型量化, 量化(使用整数而不是浮点)能够让神经网络模型运行得更快。量化是将浮点32范围的值映射为int8，同时尽量地维持模型原来的精度。
    torch_model = torch.quantization.quantize_dynamic(
        torch_model.to("cpu"), {torch.nn.Bilinear}, dtype=torch.qint8
    )
    torch_model.eval()
    model_inputs = tokenizer("元宇宙虚拟地块 430 万美元，为虚拟房产的最高成交价，为什么虚拟地块能卖这么贵？谁在为虚拟地块买单？",
                             return_tensors="pt")
    # Warm up
    input_ids, attention_mask, token_type_ids = model_inputs.data['input_ids'], model_inputs.data['token_type_ids'], \
                                                model_inputs.data['attention_mask']
    torch_out = torch_model(input_ids, attention_mask, token_type_ids)

    t0 = time()
    for _ in range(50):
        torch_out = torch_model(input_ids, attention_mask, token_type_ids)
    print('normal runtime:', time() - t0)

    torch.onnx.export(torch_model,  # model being run
                      args=(input_ids, attention_mask, token_type_ids),  # model input (or a tuple for multiple inputs)
                      f="onnx/albert_chinese_small_qu.onnx",
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization, 是否执行常量折叠优化
                      input_names=['input_ids', 'token_type_ids', 'attention_mask'],  # the model's input names
                      output_names=['output'],  # the model's output names, 按顺序分配名称到图中的输出节点
                      dynamic_axes={'input_ids': {0: 'batch_size'},  # variable lenght axes
                                    'token_type_ids': {0: 'batch_size'},
                                    'attention_mask': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


def ref_with_onnx():
    """
    使用onnxruntime,
    Returns:

    """
    model_ckpt = 'clue/albert_chinese_small'
    tokenizer = BertTokenizerFast.from_pretrained(model_ckpt)
    model_inputs = tokenizer("元宇宙虚拟地块 430 万美元，为虚拟房产的最高成交价，为什么虚拟地块能卖这么贵？谁在为虚拟地块买单？", return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

    PROVIDERS = {
        ("CPUExecutionProvider", "ONNX CPU"),
    }
    results = {}
    # ONNX
    for provider, label in PROVIDERS:
        # Create the model with the specified provider
        model = create_model_for_provider('onnx/albert_chinese_small_qu.onnx', provider)
        # Keep track of the inference time
        time_buffer = []
        # Warm up the model
        model.run(None, inputs_onnx)
        # Compute
        for _ in trange(100, desc=f"Tracking inference time on {provider}"):
            with track_infer_time(time_buffer):
                model.run(None, inputs_onnx)
        # Store the result
        results[label] = OnnxInferenceResult(
            time_buffer,
            model.get_session_options().optimized_model_filepath
        )
        print('onnxruntime runtime:')
        print(sum(results[label].model_inference_time))


# try_norm_qua()
# quantized()
gen_norm_onnx()
gen_qu_onnx()
ref_with_onnx()
