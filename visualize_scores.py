"""Reads in a scores.json file from train.py and outputs a plot of the scores."""
import json

import fire
import matplotlib.pyplot as plt
import numpy as np


def main(score_filename):
    scores = np.array(json.loads(open(score_filename).read()))
    max_scores = [np.max(x) for x in scores]

    plt.plot(np.arange(1, len(max_scores) + 1), max_scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
