epochs=1
for model in cao own zhang
do
	for dataset in SF-AIRSAR SF-RS2 OBER BRETIGNY
	do
		for dataset_method in random separate single_separated_image
		do
			for balance in none loss dataset
			do
				python principal_simulation.py --coherency --epochs $epochs --dataset_method $dataset_method --tensorflow --model $model --early_stop 50 --balance $balance --real_mode real_imag --dataset $dataset
			done
		done
	done
done
for model in cao own zhang
do
	for dataset in SF-AIRSAR SF-RS2 BRETIGNY
	do
		for dataset_method in random separate single_separated_image
		do
			for balance in none loss dataset
			do
				python principal_simulation.py --epochs $epochs --dataset_method $dataset_method --tensorflow --model $model --early_stop 50 --balance $balance --real_mode real_imag --dataset $dataset
			done
		done
	done
done
