from cvnn.data_analysis import MonteCarloAnalyzer


if __name__ == "__main__":
    hist_dict = {
        "k": [
            "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/22Wednesday/run-20h39m25/history_dict",
            "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/22Wednesday/run-13h40m03/history_dict"
        ],
        "coh": [
            "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/23Thursday/run-17h39m37/history_dict",
            "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/23Thursday/run-10h44m59/history_dict"
        ]
    }
    monte_analyzer = MonteCarloAnalyzer(history_dictionary=hist_dict)
    monte_analyzer.do_all()
