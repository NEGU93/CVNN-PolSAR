python principal_simulation.py --coherency --epochs 400 --tensorflow --model cao --early_stop 50 --real_mode real_imag --dataset OBER --dataset_method random --balance None
python principal_simulation.py --coherency --epochs 400 --tensorflow --model cao --early_stop 50 --real_mode real_imag --dataset OBER --dataset_method separate --balance None
python principal_simulation.py --coherency --epochs 400 --tensorflow --model cao --early_stop 50 --real_mode real_imag --dataset OBER --dataset_method random --balance loss
python principal_simulation.py --coherency --epochs 400 --tensorflow --model cao --early_stop 50 --real_mode real_imag --dataset OBER --dataset_method separate --balance loss
python principal_simulation.py --coherency --epochs 400 --tensorflow --model cao --early_stop 50 --real_mode real_imag --dataset OBER --dataset_method random --balance dataset
python principal_simulation.py --coherency --epochs 400 --tensorflow --model cao --early_stop 50 --real_mode real_imag --dataset OBER --dataset_method separate --balance dataset
