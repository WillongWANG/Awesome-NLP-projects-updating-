python3 -u run_seq2seq.py \
  --data_dir /home/humeng/e/projects/hot_search/modules/hot_topic/conversion_words/data/  \
  --src_file train_lucene_match_filter_with_entities_res.csv \
  --model_type unilm  \
  --model_name_or_path /home/humeng/e/resource/nlp/unilm  \
  --output_dir output_dir_cov/ \
  --log_dir log_dir_cov/ \
  --runs_dir runs_cov/ \
  --max_seq_length 30  \
  --max_position_embeddings 512 \
  --do_train  \
  --do_lower_case \
  --train_batch_size 128  \
  --learning_rate 1e-5  \
  --num_train_epochs 5