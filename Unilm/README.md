This project utilized the sequence-to-sequence language modeling capability of [UniLM](https://arxiv.org/abs/1905.03197) (via self-attention masks) trained on a generation task (source not masked, only target masked, data format: pairs of text + corresponding summaries or paraphrases) to generate abstracts or paraphrases using [Nucleus Sampling](https://arxiv.org/abs/1904.09751). The file finetune_data.csv in ```Unilm_finetuning/data/``` is used for the paraphrasing task.

![](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/p1.png)   
![](https://github.com/WillongWANG/Awesome-NLP-projects-updating-/blob/main/Unilm/p2.png)

### Pretraining:
```
python train_local.py
```
I pretrained the original model ```BertForPreTrainingLossMask from src.pytorch_pretrained_bert.modeling``` for 1000 epochs, resulting in ```model.4037.bin``` (not uploaded, try it yourself) with ```mlm_loss 7.2318 and nsp_loss 0.6929```. The performance is inferior to the provided model ```/model_dir/pytorch_model.bin``` with ```mlm_loss 6.7914 and nsp_loss 0.7013```. Even after training the model initialized from the provided ```pytorch_model.bin``` for another 3000 epochs, there was no improvement: ```mlm_loss 6.5479, nsp_loss 0.7011```. (The poor generation quality may be caused by the persistently high pre-training loss)

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

