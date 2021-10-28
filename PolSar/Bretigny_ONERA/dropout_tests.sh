
python main.py --coherency --epochs 150 --boxcar 1
python main.py --real_mode --coherency --epochs 150 --boxcar 1

python main.py --complex --coherency --epochs 150 --boxcar 5

while true
do
	python main.py --coherency --epochs 150 --boxcar 1
	python main.py --real_mode --coherency --epochs 150 --boxcar 1
	sleep 1
done

