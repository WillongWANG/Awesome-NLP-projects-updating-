This project utilized the sequence-to-sequence language modeling capability of [UniLM](https://arxiv.org/abs/1905.03197) (via self-attention masks) trained on a generation task (source not masked, only target masked, data format: pairs of text + corresponding summaries or paraphrases) to generate abstracts or paraphrases using [Nucleus Sampling](https://arxiv.org/abs/1904.09751). The file finetune_data.csv in ```Unilm_finetuning/data/``` is used for the paraphrasing task.

![](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/p1.png)   
![](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/p2.png)

### Pretraining:
```
python train_local.py
```
```/data/pretrain/pretrain_sample.txt``` contains 216830 Chinese characters (including punctuations), which can be approximately considered as 216830 tokens.  
According to [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361), model performance depends only mildly on model shape (d<sub>ff</sub>,d<sub>model</sub>,n<sub>layer</sub>,n<sub>head</sub>) with the total number of non-embedding parameters (N) fixed. For ```/bert-base-chinese/bert_config.json```(actually used in the code), the model's N is ```12*(4*768^2+4*768+2*768*3072+768+3072+2*2*768)+3*(768²+768)=86826240```.(The parameters for LayerNorm remain to be confirmed, and pooler_num_attention_heads and pooler_size_per_head are ignored)  

Based on the equation shown below, the optimal loss is around 6.5416 when not bottlenecked by compute resources, which is a high loss.
![](https://github.com/WillongWANG/Awesome-LLM-NLP-projects-updating-/blob/main/Unilm/1.png)  

Thus, I can only optimize the critical batch size as much as possible based on the equation below, which does not directly depend on model size. With our 216,830 tokens as a single batch, we could achieve a loss of around 4.1940–4.1941.  
![](https://github.com/WillongWANG/Awesome-LLM-NLP-projects-updating-/blob/main/Unilm/2.png)  

Additionally, based on the following equation, if we set parameter update steps (--total_steps) = 1000, the loss is around 4.591. If --total_steps==2000, the loss is around 3.8712.  
![](https://github.com/WillongWANG/Awesome-LLM-NLP-projects-updating-/blob/main/Unilm/3.png)  

Following the above analysis, I pretrained the original model ```BertForPreTrainingLossMask from src.pytorch_pretrained_bert.modeling``` for 1000 epochs (=total_steps because our 216,830 tokens as a single batch), resulting in ```.bin``` (not uploaded, try it yourself) with ```mlm_loss  and nsp_loss ```. The performance is inferior to the provided model ```/model_dir/pytorch_model.bin``` with ```mlm_loss 6.7914 and nsp_loss 0.7013```.(The poor generation quality may be caused by the persistently high pre-training loss)

### Fine-tuning:
```
python run_seq2seq.py
```
default parameters:  
```--num_train_epochs: 10000 #adjustable according to the results```  
```--beam_size (beam search topk): 3```

After running, replace the .bin file in ```model_dir/``` with the .bin file generated in ```output_dir/``` (renamed to ```pytorch_model.bin```). **This is important!**

Beam search and nucleus sampling demos are in ```decode_method.py```.

### Paraphrasing:
```
python decode_seq2seq.py
```

paraphrasing results are in [test.json](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/Unilm_finetuning/data/test.json).  

examples:

| test_data | top1 | top2 | top3 |
| :-------: | :--: | :--: | :--: |
| 河南濮阳一进口冷冻带鱼核酸阳性 | 冷冻带鱼，美味带鱼 | 冷冻带鱼，美味鲜美 | 冷冻带鱼，美味带鱼美味 |
| 网友呼吁抵制素媛案罪犯羽绒服牌子 | 羽绒服，穿出性感范 | 羽绒服，穿出性感感 | 羽绒服，穿出性感范儿 |
| 凡尔赛牛肉泡面 | 牛肉泡面，好吃又好吃 | 牛肉泡面，好吃又爽口 | 牛肉泡面，好吃又爽滑 |
| 迪士尼公主超绝丝绒裙 | 丝绒裙，穿出公主范 | 丝绒裙，打造公主范 | 丝绒裙，穿出公主范儿 |
| 倪妮深V锁骨杀| 深v锁骨杀，穿出女神范 | 深v锁骨杀，你的锁骨杀 | 深v锁骨杀，穿出性感锁骨杀 |

