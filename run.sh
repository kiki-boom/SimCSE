bert_checkpoint="/mnt/e/data/chinese_L-12_H-768_A-12"
#python train.py --data_dir ./NLP_Datssets --task ATEC_CCKS --model_dir ./model -bert_ckpt ${bert_checkpoint}/bert_model.ckpt --vocab_file ${bert_checkpoint}/vocab.txt --max_sequence_length 128 --batch_size 16 --epochs 3 --dropout_rate 0.3
python data_processor.py --input_dir ATEC_CCKS --output_dir ./tf_record --vocab_file ./vocab.txt --max_sequence_length 128
