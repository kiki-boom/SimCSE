python train.py \
	--data_dir ./NLP_Datssets \
	--task ATEC_CCKS \
	--model_dir ./model \
	--bert_ckpt /home/nlp/dev/base_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
	--vocab_file /home/nlp/dev/base_model/chinese_L-12_H-768_A-12/vocab.txt \
	--max_sequence_length 128 \
	--batch_size 16 \
	--epochs 3 \
	--dropout_rate 0.3
	