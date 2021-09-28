
python main.py --complex --coherency --epochs 50 --early_stop --boxcar 3
python main.py --complex --coherency --epochs 50 --early_stop --boxcar 1
python main.py --complex --coherency --epochs 50 --early_stop --boxcar 5

python main.py --complex --coherency --epochs 50 --early_stop --boxcar 3 --dropout 0.2 None 0.2
python main.py --complex --coherency --epochs 50 --early_stop --boxcar 1 --dropout 0.2 None 0.2

python main.py --complex --coherency --epochs 50 --early_stop --boxcar 3 --dropout None None 0.2
python main.py --complex --coherency --epochs 50 --early_stop --boxcar 1 --dropout None None 0.2


