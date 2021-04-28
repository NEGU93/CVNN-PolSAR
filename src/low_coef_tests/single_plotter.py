from cvnn.data_analysis import MonteCarloAnalyzer

path = "/mnt/point_de_montage/log/montecarlo/2021/04April/01Thursday/run-22h45m49"
monte = MonteCarloAnalyzer(path=path)
monte.do_all(network_filter=['complex_network', 'real_network'])
