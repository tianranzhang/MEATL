
for EPOCHS in 3 4 5 ; do 
	for LR in 2e-5 3e-5 5e-5; do
		for BATCH_SZ in 16 32 ; do
			MAX_SEQ_LEN=150

			DATA_DIR='mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0' #Modify this to be the path to the MedNLI data
			OUTPUT_DIR='mednli_output' #Modify this to be the path to your output directory
			CLINICAL_BERT_LOC=PATH/TO/CLINICAL/BERT/MODEL #Modify this to be the path to the clinical BERT model

			echo $OUTPUT_DIR
			mkdir -p $OUTPUT_DIR

		  	python3 run_classifier_ucla.py \
			  --data_dir=$DATA_DIR \
			  --bert_model='/home/tr/Desktop/MassEntailment/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000' \
			  --model_loc $CLINICAL_BERT_LOC \
			  --task_name mednli \
			  --do_train \
			  --do_eval \
			  --do_test \
			  --output_dir=$OUTPUT_DIR  \
			  --num_train_epochs $EPOCHS \
			  --learning_rate $LR \
			  --train_batch_size $BATCH_SZ \
			  --max_seq_length $MAX_SEQ_LEN \
			  --gradient_accumulation_steps 2 
		done
	done
done 