iterations=200
epochs=400
for i in `seq 2 $iterations`
do
	python3 principal_simulation.py --coherency --epochs $epochs --early_stop=50 --dataset_method random --model cao --real_mode real_imag --dataset OBER --tensorflow
	python3 principal_simulation.py --coherency --epochs $epochs --early_stop=50 --dataset_method random --model cao --dataset OBER
	python3 principal_simulation.py --coherency --epochs $epochs --early_stop=50 --dataset_method random --model cao --real_mode real_imag --dataset OBER
done

