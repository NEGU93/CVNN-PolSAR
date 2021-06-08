from cvnn.data_analysis import SeveralMonteCarloComparison, MonteCarloPlotter, MonteCarloAnalyzer
from pdb import set_trace

paths = [
    "/media/barrachina/data/results/ICASSP2021-Ober/montecarlo/2020/10October/06Tuesday/run-16h58m53",
    "/media/barrachina/data/results/MLSP2021/log/montecarlo/2021/06June/01Tuesday/run-15h31m45"
]


def find_max_epoch(path):
    monte_carlo_plotter = MonteCarloPlotter(path)
    data = monte_carlo_plotter.pandas_dict['complex network']
    maxim = data[data['stats'] == 'mean']["val_accuracy"].idxmax()
    return int(data.loc[maxim]['epoch'])


several = SeveralMonteCarloComparison('Dataset', x=['Oberpfaffenhofen', 'Bretigny'], paths=paths)
epoch = find_max_epoch(paths[1])
several.box_plot(showfig=False, savefile="./mlsp2021/results/acc_box_plot", library='seaborn', epoch=[-1, epoch])
several.box_plot(key="val_accuracy", showfig=False, savefile="./mlsp2021/results/val_acc_box_plot",
                 library='seaborn', epoch=[-1, epoch])
monte = MonteCarloAnalyzer(path=paths[1])
monte.box_plot(epoch=epoch, )
