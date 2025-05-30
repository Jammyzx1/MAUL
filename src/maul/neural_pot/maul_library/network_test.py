import os
from typing import Any, NoReturn, Union
import math
import logging
import torch
from maul.neural_pot.maul_library import network_classes
import torchani
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from maul.neural_pot.maul_library.network_utilities import (
    get_uncertainity_estimation_metrics,
    get_uncertainity_metrics,
    log_likelihood,
)
import seaborn as sns
from bayesian_torch.utils.util import predictive_entropy, mutual_information

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000


__title__ = os.path.basename(__file__)


def test(
    ani_class: network_classes.ANI,
    test_data: torchani.data.TransformableIterable,
    mc_runs: int = 2,
    plot: bool = True,
    calculate_classification_metrics: bool = False,
    plot_name_unique_prepend: str = None,
):
    """
    Function to test an ANI model from an ANI class
    :param ani_class: network_classes.ANI - ANI class
    :param test_data: torchani.data.TransformableIterable - test data
    :param mc_runs: int - n monte carlo runs to sample BNN
    :param plot: bool - plot analysis plots
    :param calculate_classification_metrics: bool - Whether to calculate classification uncertainty metrics
    :return: (list, list, float, list) -
              true_energies, predicted_energies, rmse, pred_energy_sigma
    """

    log = logging.getLogger(__name__)

    log.info("Neural potential model Test: {}".format(ani_class.model))
    true_energies, predicted_energies, rmse, pred_energy_sigma = get_energies(
        ani_class,
        test_data,
        calculate_classification_metrics,
        mc_runs=mc_runs,
    )

    if plot is True:
        plot_true_predicted_energies(true_energies, predicted_energies, rmse)

        prepend = "test"
        if plot_name_unique_prepend is not None:
            prepend = plot_name_unique_prepend + prepend

        if ani_class.isbayesian is True:
            errors = np.array(
                [p - t for p, t in zip(predicted_energies, true_energies)]
            )
            abs_errors = np.absolute(errors)

            log.info(f"Errors:\n{errors}")
            log.info(
                f"Length of predicted energies sigma: {len(pred_energy_sigma)}\nLength of errors (predicted - true): "
                f"{len(errors)}\nSaving errors, absolute errors and uncertainities to test_errors_and_uncertaininty.csv"
                f""
            )

            df = pd.DataFrame(
                data=np.array([errors, abs_errors, pred_energy_sigma]).T,
                columns=["errors", "absolute_errors", "pred_energy_sigma"],
            )
            df.to_csv("{}_errors_and_uncertaininty.csv".format(prepend), index=False)

            ax_plt = df.plot(
                x="pred_energy_sigma",
                y="errors",
                kind="scatter",
                figsize=(15, 15),
                grid=True,
                xlabel="Predicted Energy Sigma (Ha)",
                ylabel="Predicted Energy Error (predicted - true) (Ha)",
            )
            ax_plt.figure.savefig("{}_uncertainity_against_error.png".format(prepend))

            ax_plt_abs = df.plot(
                x="pred_energy_sigma",
                y="absolute_errors",
                kind="scatter",
                figsize=(15, 15),
                grid=True,
                xlabel="Predicted Energy Sigma (Ha)",
                ylabel="Predicted Absolute eEnergy Error (predicted - true) (Ha)",
            )
            ax_plt_abs.figure.savefig(
                "{}_uncertainity_against_abs_error.png".format(prepend)
            )

            uncertainty_mets = get_uncertainity_metrics(
                np.array(predicted_energies),
                np.array(pred_energy_sigma),
                np.array(true_energies),
                epoch="{}".format(prepend),
                plot=plot,
            )

            log.info("Test uncertainty metrics:\n{}".format(uncertainty_mets))

    return true_energies, predicted_energies, rmse, pred_energy_sigma


@torch.no_grad()
def get_energies(
    ani_class: network_classes.ANI,
    test_data: torchani.data.TransformableIterable,
    calculate_classification_metrics,
    mc_runs: int = 2,
):
    """
    Function to get predicted energies from a model
    :param ani_class: network_classes.ANI - ANI class
    :param test_data: torchani.data.TransformableIterable - test data
    :param mc_runs: int - n monte carlo runs to sample BNN
    :return: (list, list, float, list) -
              true_energies, predicted_energies, rmse, pred_energy_sigma
    """
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_sum = torch.nn.MSELoss(reduction="sum")
    total_mse = 0.0
    count = 0
    predicted_energy = list()
    true_energy = list()
    predicted_energy_sigma = list()

    if ani_class.isbayesian is True:

        for ith, properties in enumerate(test_data.wrapped_iterable):
            species = properties["species"].to(device)
            coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            batch_size = len(species)
            mc_predicted_energies = list()

            for mc_run in range(mc_runs):
                log.debug("----- mc run {} -----".format(mc_run))
                _, pred = ani_class.model((species, coordinates))
                log.debug("{} {}".format(mc_run, len(pred)))
                mc_predicted_energies.append(pred)

            predicted_energies = torch.stack(mc_predicted_energies).mean(axis=0)
            predicted_energies_sigma = torch.stack(mc_predicted_energies).std(axis=0)

            if calculate_classification_metrics is True:
                (
                    predictive_uncertainty,
                    model_uncertainty,
                ) = get_uncertainity_estimation_metrics(
                    torch.stack(mc_predicted_energies)
                )
                log.info(
                    f"Test set predictive uncertainty: {predictive_uncertainty} and model uncertainty: "
                    f"{model_uncertainty}"
                )

            # TODO: not sure why we build new lists like this we can just use arrays from torch
            for idx, single_species in enumerate(species):
                predicted_energy.append(predicted_energies[idx].item())
                predicted_energy_sigma.append(predicted_energies_sigma[idx].item())
                true_energy.append(true_energies[idx].item())
                total_mse += mse_sum(predicted_energies[idx], true_energies[idx]).item()
                count += 1

        rmse = hartree2kcalmol(math.sqrt(total_mse / count))

        log.info("RMSE:{} kcal/mol".format(rmse))
        return true_energy, predicted_energy, rmse, predicted_energy_sigma

    elif ani_class.isbayesian is False:

        for properties in test_data:
            species = properties["species"].to(device)
            coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            _, predicted_energies = ani_class.model((species, coordinates))

            for idx, single_species in enumerate(species):
                predicted_energy.append(predicted_energies[idx].item())
                true_energy.append(true_energies[idx].item())
                total_mse += mse_sum(predicted_energies[idx], true_energies[idx]).item()
                count += 1

        rmse = hartree2kcalmol(math.sqrt(total_mse / count))
        log.info("RMSE:{} kcal/mol".format(rmse))
        return true_energy, predicted_energy, rmse, None


def plot_true_predicted_energies(true_energies, predicted_energies, rmse):
    """
    Function to plot the true and predicted energies together
    :param true_energies: list - list of true ground truth energy values
    :param predicted_energies: list - list of predicted energy values model
    :param rmse: float - rmse value
    :return:
    """

    log = logging.getLogger(__name__)

    sns.set()

    log.info(
        "Converting true and predicted energies from Hartree to kcal/mol for test_predicted_against_known.png"
    )
    true_energies = [hartree2kcalmol(ent) for ent in true_energies]
    predicted_energies = [hartree2kcalmol(ent) for ent in predicted_energies]

    plt.figure(figsize=(5, 5))
    plt.title("Regression Plot with Loss:" + str(rmse) + " kcal/mol")
    plt.scatter(true_energies, predicted_energies, c="lightseagreen")
    sns.lineplot(x=true_energies, y=predicted_energies, label="pred_y", ci="sd")
    p1 = max(max(predicted_energies), max(true_energies))
    p2 = min(min(predicted_energies), min(true_energies))
    plt.plot([p1, p2], [p1, p2], "black", linewidth=3.0)
    plt.xlabel("True Values kcal/mol", fontsize=12)
    plt.ylabel("Predictions kcal/mol", fontsize=12)

    plt.savefig("test_predicted_against_known.png")
