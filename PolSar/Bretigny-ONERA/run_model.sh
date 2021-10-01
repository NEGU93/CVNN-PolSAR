while :
do
	python3 main.py --split_datasets --complex --epochs 150
	python3 main.py --split_datasets --complex --coherency --epochs 150
	sleep 1
done
