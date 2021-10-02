while :
do
	python3 main.py --split_datasets --epochs 150
	python3 main.py --split_datasets --coherency --epochs 150
	python3 main.py --real_mode --split_datasets --epochs 150
	python3 main.py --real_mode --split_datasets --coherency --epochs 150
	sleep 1
done
