max=20
epochs=300
for iteration in `seq 2 $max`
do
	python3 principal_simulation.py --coherency --epochs $epochs --dataset_method random --model cao --early_stop 50 --real_mode real_imag --dataset OBER
	python3 principal_simulation.py --coherency --epochs $epochs --dataset_method random --model cao --early_stop 50 --real_mode real_imag --dataset OBER --tensorflow
	python3 principal_simulation.py --coherency --epochs $epochs --dataset_method random --model cao --early_stop 50 --dataset OBER
done
