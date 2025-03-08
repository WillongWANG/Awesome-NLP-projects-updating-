#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import torch
import torch.nn.functional as F
temperature = 1.0
top_k = 0
top_p = 0.9


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    执行nucleus-sample
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) #累加

        sorted_indices_to_remove = cumulative_probs > top_p
        # 起始符号考虑进来
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]  # 用极小值填充
        logits[indices_to_remove] = filter_value
    return logits


# 模拟解码

def mock_decoder_bs(all_logits, k):
    sequences = [[list(), 1.0]]
    for cur_logit in all_logits:  # 取每个时刻的logit
        cur_logit = cur_logit / temperature
        cur_logit = F.softmax(cur_logit, dim=-1)
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]  # 记下前一个状态路径和得分
            for j in range(len(cur_logit)):  # 计算当前k个已选路径的下一时刻所有得分
                candidate = [seq + [j], score * -log(cur_logit[j])]
                all_candidates.append(candidate)
        # 所有候选根据分值排序
        ordered = sorted(all_candidates, key=lambda tup: tup[1])  # k * vocab_size
        # 选择前k个
        sequences = ordered[:k]
    return sequences


def mock_decode_nucleus(all_logits):
    sequences = [list(), 1.0]
    for cur_logit in all_logits:  # 取每个时刻的logit
        cur_logit = cur_logit / temperature
        filtered_logits = top_k_top_p_filtering(cur_logit, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token_index = torch.multinomial(probabilities, 1)  # 按照概率采样
        sequences[0] += [next_token_index.item()]
        sequences[1] *= -log(probabilities[next_token_index])  # 总概率,加-log是为了让概率是正的
    return sequences


# 定义摘要一个句子，长度为5，词典大小为10
bs_logit = torch.rand(5, 10)

# print(bs_logit)

print(mock_decode_nucleus(bs_logit))
#
# print(mock_decoder_bs(bs_logit, 3))
