from cvnn.data_analysis import SeveralMonteCarloComparison, MonteCarloAnalyzer, MonteCarloPlotter
import numpy as np


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
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/03Tuesday/run-15h43m58/run_data",      # 0.001
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/04Wednesday/run-01h35m45/run_data",    # 0.01
                                              "/media/barrachina/data/cvnn/montecarlo/2020/03March/04Wednesday/run-11h24m26/run_data",      # 0.1
                                          ], round=3)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/learning_rate/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/learning_rate/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/Simuls_29-Feb/learning_rate/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/Simuls_29-Feb/learning_rate/several_train_loss_box_plot.html")


def test_single_hidden_layer():
    several = SeveralMonteCarloComparison('correlation coefficient',
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


if __name__ == "__main__":
    # test_coef_correl()
    # test_polar_mode()
    # test_multi_class()
    # test_data_size()
    # test_learning_rate()
    # test_single_hidden_layer()
    # test_activation_function()
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/14Saturday/run-04h07m46/run_data"  # Same variance
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/14Saturday/run-20h50m08/run_data"     # Base case
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/15Sunday/run-06h44m32/run_data"       # No correl
    # path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/28Saturday/run-02h19m46/run_data"     # 4 classes
    path = "/home/barrachina/Documents/cvnn/montecarlo/2020/03March/28Saturday/run-21h28m38/run_data"    # 10 classes
    monte_carlo_analyzer = MonteCarloAnalyzer(df=None, path=path)
    monte_carlo_analyzer.do_all()
    # path = "W:/HardDiskDrive/Documentos/GitHub/cvnn/montecarlo/2020/03March/26Thursday/run-18h16m02"
    """plotter = MonteCarloPlotter(path=path)
    plotter.plot_key(key='accuracy')
    plotter.plot_key(key='accuracy', library='matplotlib')
    plotter.plot_distribution()"""
    """
    path = "W:/HardDiskDrive/Documentos/GitHub/cvnn/log/2020/03March/27Friday/run-18h47m07"
    plotter = Plotter(path=path)
    plotter.plot_key(key='accuracy')"""