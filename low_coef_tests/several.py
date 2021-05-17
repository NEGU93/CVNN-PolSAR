from cvnn.data_analysis import SeveralMonteCarloComparison

paths = [
    "./log/montecarlo/2021/04April/12Monday/run-15h12m03",
    "./log/montecarlo/2021/04April/12Monday/run-15h40m35",
    "log/montecarlo/2021/04April/12Monday/run-17h56m46",
    "log/montecarlo/2021/04April/12Monday/run-18h25m01"
]
several = SeveralMonteCarloComparison('method',
                                      x=['normal', '\"fixed\" labels', 'ordered', 'ordered + labels'],
                                      paths=paths)
several.box_plot(library='seaborn', showfig=False, key='val_accuracy', savefile='./boxplot')
several.box_plot(library='plotly', showfig=False, key='val_accuracy', savefile='./boxplot')
