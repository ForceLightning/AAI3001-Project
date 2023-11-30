"""
This script is used to generate the plots for the FGSM attack on the whitebox model.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BIM_RESULTS_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "fgsm_images")
NUM_FOLDS = 5


def main():
    sns.set("paper", style="whitegrid")

    df = pd.read_csv(os.path.join(BIM_RESULTS_PATH,
                     "metrics_results.csv"), header=0)
    ax = sns.lineplot(data=df, x="Epsilon", y="Perturbed Pixel Mean",
                      hue="Fold", legend="full")
    ax.set_xlabel(r"$L^\infty$ perturbation $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"Accuracy vs. $L^\infty$ perturbation")
    plt.savefig(os.path.join(BIM_RESULTS_PATH, "fgsm_test_acc.png"))

    ax = sns.lineplot(data=df, x="Epsilon", y="Perturbed Dice Mean",
                      hue="Fold", legend="full")
    ax.set_xlabel(r"$L^\infty$ perturbation $\varepsilon$")
    ax.set_ylabel("Dice")
    ax.set_title(r"Dice Coefficient vs. $L^\infty$ perturbation")
    plt.savefig(os.path.join(BIM_RESULTS_PATH, "fgsm_test_dice.png"))


if __name__ == "__main__":
    main()
