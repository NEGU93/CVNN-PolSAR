from cvnn.data_analysis import SeveralMonteCarloComparison, MonteCarloAnalyzer, MonteCarloPlotter
import numpy as np
from pdb import set_trace


# Simus GdR
def test_coef_correl():
    several = SeveralMonteCarloComparison('correlation coefficient',
                                          x=list(map(str, np.linspace(0, 0.707, 11)[1:])),
                                          paths=[
                                              "/media/barrachina/data/cvnn/montecarlo/2020/02February/28Friday/run-03h03m16/run_data",  # 0.1
                                              "/media/barrachina/data/cvnn/montecarlo/2020/02February/28Friday/run-12h48m12/run_data",  # 0.2
                                              "/media/barrachina/data/cvnn/montecarlo/2020/02February/28Friday/run-22h32m08/run_data",  # 0.3
                                              "/media/barrachina/data/cvnn/montecarlo/2020/02February/29Saturday/run-08h14m28/run_data",  # 0.4
                                              "/media/barrachina/data/cvnn/montecarlo/2020/02February/29Saturday/run-17h57m42/run_data",  # 0.5
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/01Sunday/run-03h45m20/run_data",  # 0.6
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/01Sunday/run-13h32m41/run_data",  # 0.7
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/01Sunday/run-23h20m08/run_data",  # 0.8
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/02Monday/run-09h09m32/run_data",  # 0.9
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/02Monday/run-19h07m26/run_data",  # 1.0
                                          ], round=2)
    several.box_plot(showfig=True, savefile="./results/Simuls_29-Feb/Coef_Correl/box_plot.html")


def test_data_size():
    mult = 0.8
    x_list = [int(mult*500), int(mult*1000), int(mult*2000), int(mult*5000), int(mult*10000)]
    several = SeveralMonteCarloComparison('data size',
                                          x=[str(x) for x in x_list],
                                          paths=[
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/20Friday/run-02h32m00/run_data",     # 500
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/20Friday/run-00h56m17/run_data",     # 1000
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/19Thursday/run-22h24m37/run_data",   # 2000
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/19Thursday/run-16h54m05/run_data",   # 5000
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/14Saturday/run-20h50m08/run_data"]   # 10000
                    )
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/data_size/test_acc_box_plot.html")
    several.box_plot(key='test loss', showfig=False, savefile="./results/Simuls_29-Feb/data_size/test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False, savefile="./results/Simuls_29-Feb/data_size/train_acc_box_plot.html")
    several.box_plot(key='train loss', showfig=False, savefile="./results/Simuls_29-Feb/data_size/box_plot.html")


def test_learning_rate():
    several = SeveralMonteCarloComparison('learning rate',
                                          x=['0.001', '0.01', '0.1'],
                                          paths=[
                                              # "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/18samedi/run-05h42m08/run_data",
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/18samedi/run-13h33m55/run_data",      # 0.001
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/12Sunday/run-22h28m23/run_data.csv",    # 0.01
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/18samedi/run-21h24m43/run_data",      # 0.1
                                          ], round=3)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/learning_rate/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/learning_rate/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/learning_rate/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/learning_rate/several_train_loss_box_plot.html")


def test_single_hidden_layer():
    several = SeveralMonteCarloComparison('Hidden Layer Size',
                                          x=["8", "16", "32","64", "128", "256", "512"],
                                          paths=["/home/barrachina/Documents/cvnn/montecarlo/2020/03March/22Sunday/run-21h34m48/run_data",       # 8
                                                 "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/22Sunday/run-13h36m19/run_data",      # 16
                                                 "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/22Sunday/run-05h39m41/run_data",      # 32
                                                 "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/21Saturday/run-21h43m44/run_data",    # 64
                                                 "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/21Saturday/run-13h48m11/run_data",    # 128
                                                 "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/21Saturday/run-05h43m56/run_data",    # 256
                                                 "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/20Friday/run-21h22m33/run_data"       # 512
                                          ])
    several.box_plot(key='test accuracy', showfig=True,
                     savefile="./results/Simuls_29-Feb/one_hidden_layer/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=True,
                     savefile="./results/Simuls_29-Feb/one_hidden_layer/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=True,
                     savefile="./results/Simuls_29-Feb/one_hidden_layer/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=True,
                     savefile="./results/Simuls_29-Feb/one_hidden_layer/several_train_loss_box_plot.html")
    several.save_pandas_csv_result(path="./results/Simuls_29-Feb/one_hidden_layer/")


def test_two_hidden_layers():
    several = SeveralMonteCarloComparison('Two hidden layer hape',
                                          x=["[32, 80]", "[32, 50]", "[64, 32]", " [85, 40]", "[128, 40]", "[256, 50]", "[512, 100]"],
                                          paths=[
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/04Saturday/run-10h35m04/run_data",    # [32, 80]
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/04Saturday/run-00h48m35/run_data",    # [32, 50]
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/03Friday/run-15h03m54/run_data",      # [64, 32]
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/03Friday/run-05h23m58/run_data",      # [85, 40]
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/14Saturday/run-20h50m08/run_data",    # [128, 40]
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/02Thursday/run-19h27m45/run_data",    # [256, 50]
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/02Thursday/run-09h04m26/run_data",    # [512, 100]
                                              ])
    several.box_plot(key='test accuracy', showfig=True,
                     savefile="./results/Simuls_29-Feb/two_hidden_layer/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=True,
                     savefile="./results/Simuls_29-Feb/two_hidden_layer/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=True,
                     savefile="./results/Simuls_29-Feb/two_hidden_layer/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=True,
                     savefile="./results/Simuls_29-Feb/two_hidden_layer/several_train_loss_box_plot.html")
    several.save_pandas_csv_result(path="./results/Simuls_29-Feb/two_hidden_layer/")


def test_activation_function():
    several = SeveralMonteCarloComparison('activation function',
                                          x=['ReLU', 'sigmoid', 'tanh', 'Leaky ReLU'],
                                          paths=[
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/14Saturday/run-20h50m08/run_data",   # ReLU
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/18Wednesday/run-11h08m20/run_data",  # sigmoid
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/18Wednesday/run-21h00m52/run_data",  # tanh
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/19Thursday/run-06h54m34/run_data"    # Leaky ReLU
                                          ])
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/activation_function/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/activation_function/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/activation_function/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/activation_function/several_train_loss_box_plot.html")


def test_polar_mode():
    several = SeveralMonteCarloComparison('polar mode',
                                          x=['same variance', 'no correlation'],
                                          paths=[
                                              # "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/26Thursday/run-19h46m15/run_data",     # Base
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/27Friday/run-05h56m03/run_data",         # Same Variance
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/27Friday/run-16h08m23/run_data"          # No coef correl
                                          ])
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/polar_mode/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/polar_mode/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/polar_mode/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/polar_mode/several_train_loss_box_plot.html")


def test_polar_mode_one_layer():
    several = SeveralMonteCarloComparison('polar mode',
                                          x=['Type A', 'Type B'],
                                          paths=[
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/13Monday/run-15h46m21/run_data",
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/13lundi/run-23h39m30/run_data"
                                          ])
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/polar_mode_one_layer/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/polar_mode_one_layer/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/polar_mode_one_layer/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/polar_mode_one_layer/several_train_loss_box_plot.html")


def test_multi_class():
    several = SeveralMonteCarloComparison('Multi-class',
                                          x=['4', '10'],
                                          paths=[
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/28Saturday/run-02h19m46/run_data",   # 4
                                              "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/28Saturday/run-21h28m38/run_data"      # 10
                                          ])
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/multi_class/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/multi_class/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/multi_class/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/multi_class/several_train_loss_box_plot.html")

# Simus MLSP


def test_data_size_mlsp():
    mult = 1
    x_list = [int(mult*1000), int(mult*2000), int(mult*5000), int(mult*10000)]
    several = SeveralMonteCarloComparison('data size',
                                          x=[str(x) for x in x_list],
                                          paths=[
                                              # "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/09jeudi/run-19h08m30/run_data",    # 500
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/09jeudi/run-05h05m05/run_data",     # 1000
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/08mercredi/run-17h23m32/run_data",   # 2000
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/08mercredi/run-06h57m40/run_data",     # 5000
                                              "/media/barrachina/data/cvnn/MLSP/montecarlo/2020/03March/14Saturday/run-20h50m08/run_data"   # 10000
                                          ]
                    )
    several.box_plot(library='seaborn', savefile="./results/dataset_box_plot", showfig=True)
    """several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/data_size/test_acc_box_plot.html")
    several.box_plot(key='test loss', showfig=False, savefile="./results/data_size/test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False, savefile="./results/data_size/train_acc_box_plot.html")
    several.box_plot(key='train loss', showfig=False, savefile="./results/data_size/train_loss_box_plot.html")"""


def test_coef_correl_1hl():
    several = SeveralMonteCarloComparison('correlation coefficient',
                                          x=list(map(str, [0.1, 0.2, 0.35, 0.5, 0.65, 0.85, 0.99])),
                                          paths=[
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/15Wednesday/run-07h31m01/run_data",    # 0.1
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/15mercredi/run-15h21m20/run_data",      # 0.2
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/15mercredi/run-23h08m45/run_data",     # 0.35
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/12Sunday/run-22h28m23/run_data",       # 0.5
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/16jeudi/run-06h50m33/run_data",        # 0.65
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/16jeudi/run-14h35m51/run_data",        # 0.85
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/16jeudi/run-22h23m11/run_data",        # 0.99
                                          ], round=2)
    several.box_plot(showfig=False, savefile="./results/Coef_Correl_1HL/box_plot", library='seaborn')


def test_coef_correl_1hl_dropout():
    several = SeveralMonteCarloComparison('correlation coefficient',
                                          x=list(map(str, [0.1, 0.2, 0.35, 0.5, 0.7, 0.99])),
                                          paths=[
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/23Thursday/run-18h15m23/run_data",     # 0.1
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/25Saturday/run-14h32m15/run_data",     # 0.2
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/26Sunday/run-13h02m33/run_data",       # 0.35
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/22Wednesday/run-14h55m36/run_data",    # 0.5
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/26dimanche/run-22h07m29/run_data",     # 0.7
                                              "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04avril/27lundi/run-07h06m17/run_data"         # 0.99
                                          ], round=2)
    several.box_plot(showfig=False, savefile="./results/Coef_Correl_1HL_drop/box_plot", library='seaborn')


if __name__ == "__main__":
    # test_coef_correl_1hl_dropout()
    # test_data_size_mlsp()
    # test_two_hidden_layers()
    # test_coef_correl()
    # test_polar_mode()
    # test_multi_class()
    # test_data_size()
    # test_data_size_mlsp()
    # test_learning_rate()
    # test_single_hidden_layer()
    # test_activation_function()
    # test_polar_mode_one_layer()
    # path = "/media/barrachina/data/cvnn/MLSP/montecarlo/2020/03March/14Saturday/run-04h07m46/run_data"  # Same variance
    # path = "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/12Sunday/run-22h28m23/run_data.csv"   # Base with 1 layer
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/14Saturday/run-20h50m08/run_data"     # Base case
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/15Sunday/run-06h44m32/run_data"       # No correl
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/28Saturday/run-02h19m46/run_data"     # 4 classes
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/28Saturday/run-21h28m38/run_data"    # 10 classes
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/04April/01Wednesday/run-17h47m52/run_data"  # New Sigmoid
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/21Saturday/run-21h43m44/run_data"   # one hidden 64
    # path = "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/06Monday/run-11h08m09/run_data"    # Both variance and coef correl
    # path = "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/06Monday/run-20h57m46/run_data"   # tanh but with one layer only
    # monte_carlo_analyzer = MonteCarloAnalyzer(df=None, path=path)
    # monte_carlo_analyzer.do_all()
    # monte_carlo_analyzer.plot_histogram(library='matplotlib', extension=".png")
    # path = "W:/HardDiskDrive/Documentos/GitHub/cvnn/montecarlo/2020/03March/26Thursday/run-18h16m02"
    """plotter = MonteCarloPlotter(path=path)
    plotter.plot_key(key='accuracy')
    plotter.plot_key(key='accuracy', library='matplotlib')
    plotter.plot_distribution()"""
    """
    path = "W:/HardDiskDrive/Documentos/GitHub/cvnn/log/2020/03March/27Friday/run-18h47m07"
    plotter = Plotter(path=path)
    plotter.plot_key(key='accuracy')"""
    # path = "/home/barrachina/Documents/onera/log/montecarlo/2020/07July/29Wednesday/run-14h37m01/run_data"
    # path = "/home/barrachina/Documents/onera/log/montecarlo/2020/07July/30Thursday/run-16h45m09/run_data.csv"
    # monte_carlo_analyzer = MonteCarloAnalyzer(df=None, path=path)
    # monte_carlo_analyzer.box_plot(library='seaborn')
    paths = [
        "log/montecarlo/2020/08August/30Sunday/run-03h06m08/run_data",
        "log/montecarlo/2020/08August/30Sunday/run-14h35m58/run_data"
    ]
    x = [
        "cartesian",
        'polar'
    ]
    several_monte_carlo = SeveralMonteCarloComparison(label="Polar", x=x, paths=paths)
    # several_monte_carlo.box_plot(showfig=True)
    several_monte_carlo.monte_carlo_analyzer.df = several_monte_carlo.monte_carlo_analyzer.df[
        several_monte_carlo.monte_carlo_analyzer.df['network'] != 'complex network polar']
    several_monte_carlo.plot_histogram(savefig=True, showfig=False, library='seaborn')
    # set_trace()
