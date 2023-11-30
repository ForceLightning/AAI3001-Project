"""
This script is used to generate the plots for the BIM attack on the whitebox model.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BIM_RESULTS_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "bim_images")
NUM_FOLDS = 5


def main():
    sns.set("paper", style="whitegrid")

    df = pd.read_csv(os.path.join(BIM_RESULTS_PATH, "results.csv"), header=0)
    ax = sns.lineplot(data=df, x="Epsilon", y="Perturbed Pixel Mean",
                      hue="Fold", style="Alpha", legend="full")
    ax.set_xlabel(r"$L^\infty$ perturbation $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"Accuracy vs. $L^\infty$ perturbation")
    plt.savefig(os.path.join(BIM_RESULTS_PATH, "bim_test_acc.png"))

    ax = sns.lineplot(data=df, x="Epsilon", y="Perturbed Dice Mean",
                      hue="Fold", style="Alpha", legend="full")
    ax.set_xlabel(r"$L^\infty$ perturbation $\varepsilon$")
    ax.set_ylabel("Dice")
    ax.set_title(r"Dice Coefficient vs. $L^\infty$ perturbation")
    plt.savefig(os.path.join(BIM_RESULTS_PATH, "bim_test_dice.png"))


if __name__ == "__main__":
    main()
