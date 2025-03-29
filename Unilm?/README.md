This project utilized the sequence-to-sequence language modeling capability of [UniLM](https://arxiv.org/abs/1905.03197) (via self-attention masks) trained on a generation task (source not masked, only target masked, data format: pairs of text + corresponding summaries or paraphrases) to generate abstracts or paraphrases using [Nucleus Sampling](https://arxiv.org/abs/1904.09751). 
The file finetune_data.csv in ```Unilm_finetuning/data/``` is used for the paraphrasing task.

![]([https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/)p1.png)   
![]([https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/)p2.png)

### Pretraining:
```
python train_local.py
```
**Some empirical analysis:**    
```/data/pretrain/pretrain_sample.txt``` contains 216830 Chinese characters (including punctuations), which can be approximately considered as 216830 tokens.  
According to [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361), model performance depends only mildly on model shape (d<sub>ff</sub>,d<sub>model</sub>,n<sub>layer</sub>,n<sub>head</sub>) with the total number of non-embedding parameters (N) fixed. 
For ```/bert-base-chinese/bert_config_uncased_tiny.json```, the model's N is ```2*(4*32^2+4*32+2*32*32+2*32+2*2*32)=12928```, where our 216830 tokens does not satisfy the following equation (data size is not enough), so overfitting may not be avoided.  
![](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/4.png)  
```bert_config.json```'s N is ```12*(4*768^2+4*768+2*768*3072+768+3072+2*2*768)+3*(768²+768)=86826240```.(The parameters for LayerNorm remain to be confirmed, and pooler_num_attention_heads and pooler_size_per_head are ignored)  
Based on the equation shown below, the optimal loss is around 6.6489 with N=12928 when not bottlenecked by compute resources, which is a high loss.  
![](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/1.png)  
Thus, we can only optimize the critical batch size as much as possible based on the equation below, which does not directly depend on model size. With our 216,830 tokens as a single batch, we could achieve a loss of around 4.1940–4.1941. However, the problem of overfitting cannot be solved.  
![](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/2.png)  
Additionally, based on the following equation, the first term alone already reaches approximately 5.5848 with N=12928, which is still a relatively high loss.  
![](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/3.png)  

### Fine-tuning:
```
python run_seq2seq.py
```
default parameters:  
```--beam_size (beam search topk): 3```

After running, replace the .bin file in ```model_dir/``` with the .bin file generated in ```output_dir/``` (renamed to ```pytorch_model.bin```). **This is important!**

Beam search and nucleus sampling demos are in ```decode_method.py```.

### Paraphrasing:
```
python decode_seq2seq.py
```

paraphrasing results are in [test.json](https://github.com/WillongWang/Awesome-LLM-NLP-projects-updating-/edit/main/Unilm%3F/Unilm_finetuning/data/test.json).  

examples:

| test_data | top1 | top2 | top3 |
| :-------: | :--: | :--: | :--: |
| 河南濮阳一进口冷冻带鱼核酸阳性 | 冷冻带鱼，美味带鱼 | 冷冻带鱼，美味鲜美 | 冷冻带鱼，美味带鱼美味 |
| 网友呼吁抵制素媛案罪犯羽绒服牌子 | 羽绒服，穿出性感范 | 羽绒服，穿出性感感 | 羽绒服，穿出性感范儿 |
| 凡尔赛牛肉泡面 | 牛肉泡面，好吃又好吃 | 牛肉泡面，好吃又爽口 | 牛肉泡面，好吃又爽滑 |
| 迪士尼公主超绝丝绒裙 | 丝绒裙，穿出公主范 | 丝绒裙，打造公主范 | 丝绒裙，穿出公主范儿 |
| 倪妮深V锁骨杀| 深v锁骨杀，穿出女神范 | 深v锁骨杀，你的锁骨杀 | 深v锁骨杀，穿出性感锁骨杀 |

