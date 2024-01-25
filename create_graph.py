from libmultilabel.linear.utils import load_pipeline
from experiment_utils import parse_folder, get_size
import matplotlib.pyplot as plt
import numpy as np
import os
from threshold_experiment import load_config

def histogram(data, nbin, mean, median):
    config = load_config("experiment_config.yaml")
    experiment_name_str = str(config["experiment_name"])
    print("Mean: ", mean)
    print("Median: ", median)
    plt.hist(data, nbin, histtype="bar", rwidth=0.5, log=True)
    plt.axvline(x=np.median(data), color="r", label="axvline - full height")
    plt.title(config["experiment_name"] + " Weights Distribution")
    plt.xlabel("weights")
    plt.ylabel("frequency")
    plt.savefig(
        os.path.join(
            config["results_folder_path"], config["experiment_name"], f"{experiment_name_str}_weights_distribution.png"
        )
    )
    plt.close()

# rename x axis to abs value of weight distribution
# double check original graph and the spikes in values
if __name__ == "__main__":
    config = load_config("experiment_config.yaml")

    try:
        os.mkdir(os.path.join("experiment_results", config["experiment_name"]))
    except OSError as error:
        pass

    result_path = os.path.join("experiment_results", config["experiment_name"])

    preprocessor, model = load_pipeline(os.path.join(config["baseline_model_dir"], "linear_pipeline.pickle"))
    sparse_data = np.abs(model.flat_model.weights.data)

    # Histogram representing the weight distribtuion of the baseline model
    histogram(sparse_data, nbin=100, mean=np.mean(sparse_data), median=np.median(sparse_data))

    # Parse experiment data
    size_dict = get_size(os.path.join(config["baseline_model_dir"], config["experiment_folder_name"]))
    res = parse_folder("runs/", size_dict)

    experiment_logs_path = os.path.join("experiment_logs", config["experiment_name"])
    max_val = 0
    min_val = 100
    for item in res:
        max_val = max(max_val, float(item[0]))
        min_val = min(min_val, float(item[0]))

    experiment_name_str = str(config["experiment_name"])
    res = np.array(res)
    plt.scatter(res[:, 0], res[:, 1], marker=".")
    plt.scatter(res[:, 0], res[:, 2], marker=".")
    plt.scatter(res[:, 0], res[:, 3], marker=".")
    plt.title(config["experiment_name"] + " Threshold vs P@K-values")
    # plt.xlim(-0.00000000001,.00000005)
    # plt.plot(np.median(sparse_data), 0, np.median(sparse_data), marker='o')
    plt.xscale("linear")
    plt.xlabel("threshold")
    plt.ylabel("scores")

    # plt.axvline(x = np.median(sparse_data), color = 'b', label = 'axvline - full height')
    plt.legend(["p1", "p3", "p5"])
    plt.savefig(os.path.join(result_path, f"{experiment_name_str}_threshold_vs_prediction.png"))
    # plt.savefig(result_path + "/" + config['experiment_name'] + "_threshold_vs_prediction.png")
    plt.close()

    plt.scatter(list(res[:, -1]), list(res[:, 1]), s=1)
    plt.scatter(list(res[:, -1]), list(res[:, 2]), s=1)
    plt.scatter(list(res[:, -1]), list(res[:, 3]), s=1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("file size (bytes)")
    plt.title(config["experiment_name"] + " Filesize vs P@K Values")
    plt.ylabel("scores")
    plt.legend(["p1", "p3", "p5", "median"])
    plt.savefig(os.path.join(result_path, f"{experiment_name_str}_file_size_vs_prediction.png"))
    # plt.savefig(result_path + f"/{config["experiment_name"]}_file_size_vs_prediction.png")
    plt.close()

    plt.scatter(list(res[:, 0]), list(res[:, -1]), s=20, marker="*")
    plt.xlabel("threshold")
    plt.ylabel("file size (bytes)")
    plt.title(config["experiment_name"] + " Filesize vs Thresholds")
    # plt.ylim(0,)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(os.path.join(result_path, f"{experiment_name_str}_file_size_vs_threshold.png"))
    # plt.savefig(result_path + f"/{config["experiment_name"]}_file_size_vs_threshold.png")
    plt.close()
