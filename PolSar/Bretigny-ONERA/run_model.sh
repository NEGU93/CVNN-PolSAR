while :
do
	python3 main.py --split_datasets --epochs 200
	python3 main.py --split_datasets --coherency --epochs 200
	python3 main.py --real_mode --split_datasets --epochs 200
	python3 main.py --real_mode --split_datasets --coherency --epochs 200
	sleep 1
done
