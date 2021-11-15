epochs=100
for i in {1..50};
do
		python3 principal_simulation.py --epochs $epochs --dataset_method random --model cao --balance none --dataset BRET
done

