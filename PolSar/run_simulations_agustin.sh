epochs=70
for i in {1..50};
do
	python principal_simulation.py --epochs $epochs --dataset_method random --model cao --balance none --dataset BRET
done

