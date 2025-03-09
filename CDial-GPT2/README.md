This project utilizes the [CDial-GPT2_LCCC-base](https://huggingface.co/thu-coai/CDial-GPT2_LCCC-base) Chinese dialogue pre-training model, fine-tuned on the large-scale Chinese dialogue dataset [STC](https://arxiv.org/abs/1503.02364)，and compares it with the [original project](https://github.com/thu-coai/CDial-GPT/tree/master?tab=readme-ov-file).

![](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/CDial-GPT2/figures/inputs.png)

## Requirements
torch==1.4.0  
(python<3.9, or pip can not find :D)  
torchvision==0.5.0  
pytorch-ignite==0.2.1  
transformers==2.1.1  
tensorboardX==1.8  
protobuf==3.20.3  

## Fine-tuning

### Dataset
download LCCC-base to ./data

### Model
I recommend to download it yourself from [huggingface](https://huggingface.co/thu-coai/CDial-GPT2_LCCC-base/tree/main) to avoid HTTP error, and create the same directory thu-coai/CDial-GPT2_LCCC-base in your working directory. 

### Run
```
python train.py --pretrained --gpt2 --model_checkpoint thu-coai/CDial-GPT2_LCCC-base --data_path data/STC.json --scheduler linear
#very slow on only one GPU, even few weeks for only 1 epoch on STC including over 4 million pairs 
```
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --gpt2 --pretrained --model_checkpoint thu-coai/CDial-GPT2_LCCC-base --data_path data/STC.json --scheduler linear
#Training on 4 GPUs distributedly, following the original paper(4*RTX 2080 Ti)
```
In the original paper, the number of the warmup epoch was set to 1, and the maximum learning rate was 6.25e-5. The batch size was set to 8, fine-tune epochs was 10. But in the original code, lr=5e-5, batch size=2, warmup steps=5000, I follow the same settings.   
The model was trained on 4*RTX 2080 Ti at AutoDL platform (nearly 30h for only around 1 epoch (>_<)) as an RTX 3080 or higher GPU requires CUDA version 11.x or above to be utilized.

![](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/CDial-GPT2/p.png)

### Inference
After running train.py, remember to load the model parameters from the .pth checkpoint file in the runs/ directory.
You can check it yourself:
```
import torch
from transformers import GPT2LMHeadModel
ckpt = torch.load("YOUR_pth", map_location="cpu")
print(ckpt.keys())
#for example
model = GPT2LMHeadModel.from_pretrained("thu-coai/CDial-GPT2_LCCC-base")
print("Before loading:", model.transformer.wte.weight[0, :5])
state_dict = torch.load("YOUR_pth", map_location="cpu")
model.load_state_dict(state_dict)
print("After loading:", model.transformer.wte.weight[0, :5])
```
Infer:
```
YOUR_MODEL_PATH: : the model path used for generation (i.e. fine-tuned model:.../model_training_args.bin)
python infer.py --gpt2 --model_checkpoint YOUR_MODEL_PATH --datapath data/STC_test.json --out_path STC_result.txt  # Inference on a corpus
python interact.py --gpt2 --model_checkpoint YOUR_MODEL_PATH  # Interact on the terminal
```

### Metrics

| model (all fine-tuned on STC) | size | PPL | BLEU-2 | BLEU-4 | Dist-1 | Dist-2 | Greedy Matching | Embedding Average |
| :------ | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Attn-Seq2seq | 73M | 34.20 | 3.93 | 0.90 | 8.5 | 11.91 | 65.84 | 83.38 |
| Transformer | 113M | 22.10 | 6.72 | 3.14 | 8.8 | 13.97 | 66.06 | 83.55 |
| [GPT2-chitchat (baseline)](https://github.com/yangjianxin1/GPT2-chitchat) | 88M | - | 2.28 | 0.54 | 10.3 | 16.25 | 61.54 | 78.94 |
| GPT<sub>Novel</sub> | 95.5M | 21.27 | 5.96 | 2.71 | 8.0 | 11.72 | 66.12 | 83.34 |
| GPT<sub>LCCC-base</sub> | 95.5M | 18.38 | 6.48 | 3.08 | 8.3 | 12.68 | 66.21 | 83.54 |
| GPT2<sub>LCCC-base</sub> | 95.5M | 22.76 | 5.69 | 2.50 | 7.7 | 10.87 | 66.24 | 83.46 |
| GPT<sub>LCCC-large</sub> | 95.5M | 18.23 | 6.63 | 3.20 | 8.3 | 12.71 | 66.25 | 83.63 |

In the orginal paper, GPT2 and GPT have the same size and been trained on the same dataset, while GPT2 performs worse. I don't understand...

## Generated dialogues

The original results (single-turn generated examples from STC dataset using GPT<sub>LCCC-large</sub> fine-tuned on STC)：

| Post | Response |
| :----- | :---------- |
| 这 居 然 是 北 京 不 是 哈 尔 滨 。 | 哈 尔 滨 的 天 气 好 像 比 北 京 好 点 ， 我 这 里 就 是 小 雨 ， 白 天 都 是 大 风 |
| 运 营 商 校 园 广 告 大 P K ， 太 欢 乐 了 ！ 哈 哈 哈 。 | 我 以 为 是 马 云 的 广 告 。 |
| D e l s a S p o s a 糖 果 色 系 列 婚 纱 ， 粉 蓝 红 紫 ， 俏 皮 又 清 新 ， 你 喜 欢 吗 ？ | 最 后 一 件 太 美 了 ， 美 的 不 像 话 ！ |

interactive script：

| Role | Response |
| :----- | :---------- |
| User | 别爱我没结果 |
| Bot | 没 结 果 的 爱 是 多 么 的 痛 。 |
| User | 但是爱过就可以了 |
| Bot | 有 时 候 ，爱 情 就 是 一 场 游 戏 。 |
| User | 输的总是我 |
| Bot | 我 爱 你 ， 就 像 我 爱 你 一 样 。 |
| User | 你可拉到吧 |
| Bot | 你 可 拉 到 吧 。 |
| User | 别学我说话 |
| Bot | 我 说 话 就 是 你 不 行 。 |
| User | 逼我动手是吧 |
| Bot | 不 ， 是 逼 你 动 口 是 吧 |

<br>
  
My results (CDial-GPT2_LCCC-base fine-tuned on STC):

| Post | Response |
| :----- | :---------- |
| 这 居 然 是 北 京 不 是 哈 尔 滨 。 | 天 津 也 有 这 么 大 的 雪 ， 没 有 下 雪 |
| 运 营 商 校 园 广 告 大 P K ， 太 欢 乐 了 ！ 哈 哈 哈 。 | 好 棒 啊 ！ 有 机 会 一 定 要 去 看 看 |
| D e l s a S p o s a 糖 果 色 系 列 婚 纱 ， 粉 蓝 红 紫 ， 俏 皮 又 清 新 ， 你 喜 欢 吗 ？ | 好 漂 亮 的 红 色 ， 可 惜 我 不 喜 欢 。 |

interactive script：

| Role | Response |
| :----- | :---------- |
| User | 别爱我没结果 |
| Bot | 别 爱 我 了 ， 一 起 死 磕 |
| User | 但是爱过就可以了 |
| Bot | 别 爱 我 没 结 果 就 可 以 了 |
| User | 输的总是我 |
| Bot | 没 结 果 ， 一 起 死 磕 |
| User | 你可拉到吧 |
| Bot | 输 的 总 是 我 |
| User | 别学我说话 |
| Bot | 别 爱 我 了 ， 我 都 不 爱 你 了 |
| User | 逼我动手是吧 |
| Bot | 我 说 的 是 别 人 |

hhh...

<br>
   
or just use the original CDial-GPT2_LCCC-base:

| Post | Response |
| :----- | :---------- |
| 这 居 然 是 北 京 不 是 哈 尔 滨 。 | 你 咋 知 道 |
| 运 营 商 校 园 广 告 大 P K ， 太 欢 乐 了 ！ 哈 哈 哈 。 | 你 不 是 说 我 么 |
| D e l s a S p o s a 糖 果 色 系 列 婚 纱 ， 粉 蓝 红 紫 ， 俏 皮 又 清 新 ， 你 喜 欢 吗 ？ | 我 喜 欢 ！ 哈 哈 哈 ！ |

interactive script：

| Role | Response |
| :----- | :---------- |
| User | 别爱我没结果 |
| Bot | 没 有 |
| User | 但是爱过就可以了 |
| Bot | 我 也 是 |
| User | 输的总是我 |
| Bot | 哈 哈 |
| User | 你可拉到吧 |
| Bot | 可 以 啊 |
| User | 别学我说话 |
| Bot | 好 吧 ， 我 以 为 你 说 话 呢 |
| User | 逼我动手是吧 |
| Bot | 我 不 要 你 动 手 就 行 了 |

em...somewhat perfunctory

