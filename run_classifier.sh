
for lambd in 0.2; do
	for LR_P in 1e-5; do
		for LR_Q in 5e-4 ; do
			for nc in 1; do
				for target_dir in '../data/mimic/'; do #'../data/mimic/bootstrap1/' '../data/mimic/bootstrap2/' '../data/mimic/bootstrap3/' '../data/mimic/bootstrap4/' '../data/mimic/bootstrap5/'; do #'../data/mimic/'; do #
                
				    python3 train_ATN.py \
                    --F_layers=2\
                    --P_layers=0\
                    --Q_layers=2\
                    --Q_learning_rate $LR_Q\
                    --attn='dot'\
                    --batch_size=6\
                    --ch_train_lines=0\
                    --clip_lower=-0.01\
                    --clip_upper=0.01\
                    --dropout=0.2\
                    --emb_filename='../data/ucla_mimic_emb/emb_aligned_with_mimic_loinc.txt'\
                    --emb_size=768\
                    --en_train_lines=0\
                    --head_num=12\
                    --hidden_size=768\
                    --kernel_num=400\
                    --lambd $lambd\
                    --learning_rate=$LR_P\
                    --max_epoch=20\
                    --max_seq_len=1024\
                    --model='transformer'\
                    --model_save_file='../param_search/nc/allucla_loinc_F2_P0_Q2_max_len1024_adamw_plr_0.00001_qlr0.0005_lambd'${lambd}'nc'${nc}'_warmup500_3122/run1'\
                    --n_critic $nc\
                    --num_warmup_steps_Q=3122\
                    --random_seed=1\
                    --src_data_dir='../data/UCLA_labs/'\
                    --t_hidden_size=768\
                    --tgt_data_dir $target_dir
                done
			done
		done
	done
done 