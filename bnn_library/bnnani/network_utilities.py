import logging
from typing import Any, NoReturn, Union
import os
import torchani
import numpy as np
import torch
from bayesian_torch.utils.util import predictive_entropy, mutual_information
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from torchani.units import hartree2kcalmol
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import re

__title__ = os.path.basename(__file__)

def get_uncertainity_estimation_metrics(predictions):
    """
    Function to get metrics on the uncertainty
    :param predictions: torch.tensor - predictions from the model
    :return: (float, float) - predictive uncertainty model uncertainty
    """

    log = logging.getLogger(__name__)

    predictions = predictions.data.cpu().numpy()

    log.debug("Getting uncertainty estimation metrics")
    log.debug("Values:\n{}".format(predictions))

    predictive_uncertainty = predictive_entropy(predictions)
    model_uncertainty = mutual_information(predictions)

    return predictive_uncertainty, model_uncertainty

def get_means(model, ensemble_length=8, to_list=False):
    """
    Function to calculate the means over weights and biases of an ensemble of networks
    :param model: pytorch model
    :param ensemble_length: int - number of models in the ensemble
    :param to_list: bool - Whether to return the standard deviations as lists or tensors
    :return: (dict, dict) - standard deviations of the weight, standard deviations of the biases
    """

    log = logging.getLogger(__name__)

    mean_weights = {}
    mean_biases = {}

    log.info(f"Mean normalization is {ensemble_length} (this is should be equal to the number of models)")

    for ith in range(0, ensemble_length):
        log.debug(f"Network {ith}\n")
        for name, param in model[ith].named_parameters():

            log.debug(f"Name: {name} Layer contains {param.size()} parameters.")
            log.debug(f"{param}\n")

            if "weight" in name:

                if ith == 0:
                    mean_weights[name] = torch.clone(param)

                elif ith == ensemble_length-1:
                    mean_weights[name] = mean_weights[name] + param
                    mean_weights[name] = mean_weights[name] / len(model)
                    if to_list is True:
                        mean_weights[name] = mean_weights[name].cpu().detach().tolist()

                else:
                    mean_weights[name] = mean_weights[name] + param

            elif "bias" in name:

                if ith == 0:
                    mean_biases[name] = torch.clone(param)

                elif ith == ensemble_length-1:
                    mean_biases[name] = mean_biases[name] + param
                    mean_biases[name] = mean_biases[name] / len(model)
                    if to_list is True:
                        mean_biases[name] = mean_biases[name].cpu().detach().tolist()

                else:
                    mean_biases[name] = mean_biases[name] + param

    return mean_weights, mean_biases


def get_std(model, ensemble_length=8, to_list=False):
    """
    Function to calculate the standard deviation over weights and biases of an ensemble of networks
    :param model: pytorch model
    :param ensemble_length: int - number of models in the ensemble
    :param to_list: bool - Whether to return the standard deviations as lists or tensors
    :return: (dict, dict) - standard deviations of the weight, standard deviations of the biases
    """

    log = logging.getLogger(__name__)

    std_parameters_weights = {}
    std_parameters_biases = {}

    log.info(f"Mean normalization is {ensemble_length} (this is should be equal to the number of models)")

    for ith in range(0, ensemble_length):
        log.debug(f"Network {ith}\n")
        for name, param in model[ith].named_parameters():

            log.debug(f"Name: {name} Layer contains {param.size()} parameters.")
            log.debug(f"{param}\n")

            if "weight" in name:

                if ith == 0:
                    std_parameters_weights[name] = torch.empty(param.size()[0], param.size()[1], len(model))
                    std_parameters_weights[name][:, :, ith] = torch.clone(param)

                elif ith == ensemble_length - 1:
                    std_parameters_weights[name][:, :, ith] = torch.clone(param)
                    std_parameters_weights[name] = torch.std(std_parameters_weights[name], 2, False)
                    if to_list is True:
                        std_parameters_weights[name] = std_parameters_weights[name].cpu().detach().tolist()

                else:
                    std_parameters_weights[name][:, :, ith] = torch.clone(param)

            elif "bias" in name:

                if ith == 0:
                    std_parameters_biases[name] = torch.empty(param.size()[0], len(model))
                    std_parameters_biases[name][:, ith] = torch.clone(param)


                elif ith == ensemble_length - 1:
                    std_parameters_biases[name][:, ith] = torch.clone(param)
                    std_parameters_biases[name] = torch.std(std_parameters_biases[name], 1, False)
                    if to_list is True:
                        std_parameters_biases[name] = std_parameters_biases[name].cpu().detach().tolist()

                else:
                    std_parameters_biases[name][:, ith] = torch.clone(param)

            else:
                log.info(f"Unknown parameter {name} given will ignore please check this.")

    return std_parameters_weights, std_parameters_biases


def get_rho_from_sigma(sigma):
    """
    Function to convert standard deviation to variation rho values
    :param sigma: torch.tensor - tensor of standard deviation values
    """

    rho = torch.log(torch.expm1(torch.abs(sigma)) + 1e-20)
    return rho

def log_likelihood(prediction: float,
                 standard_deviation: float,
                 true_value:float):
    """
    A function to compute the log likelihood for an output. Assumes that the PDF is Gaussian.

    f(x|u, s) = ln(sqrt(2*pi*s^(2))) - (x - u)^(2)
                                       -----------
                                         2s^(2)

    x is the data point known value
    u is the mean
    s is the standard deviation

    :param prediction: float - predicted values from the model
    :param standard_deviation: float - standard deviation of the prediction
    :param true_value: float - the known value of the predicted
    :return: float
    """

    return np.log(2.0 * np.pi * standard_deviation**2) - ((true_value - prediction)**2 / (2.0 * (standard_deviation**2)))

def root_mean_variance(predictions_stdev):
    """
    A measure for the variance from https://arxiv.org/pdf/2207.06916.pdf
         _________________________________
        /
      \/  1/N * sum(standard deviation ^2)
    :param predictions_stdev: iterable - lits/array of floats
    :return: float
    """
    if isinstance(torch.tensor, predictions_stdev):
        sigmas = predictions_stdev.cpu().detach().numpy()
    else:
        sigmas = predictions_stdev

    rmv = np.sqrt(1/len(sigmas) * (sum([s*s for s in sigmas])))

    return rmv

def get_uncertainity_metrics(predictions: np.ndarray,
                 standard_deviation: np.ndarray,
                 true_value: np.ndarray,
                 x: np.ndarray = None,
                 verbose: bool = False,
                 number_of_bins: int = 100,
                 plot: bool = True,
                 epoch: Union[int, str, None] = None,
                 max_sample: int = 2000,
                 plot_summary: bool = True) -> dict:
    """
    Use uncertainty-toolbox package to get uncertainty metric data and plots
    :param predictions: np.ndarray - iterable array of predicted values
    :param standard_deviation: np.ndarray - iterable array of standard deviations over the predictions
    :param true_value: np.ndarray - iterable array of true values
    :param x: np.ndarray - inputs for the predictions is None it will be a zero based index
    :param verbose: bool - print metrics or not
    :param number_of_bins: int - number of bins to use for discretized metrics
    :param plot: bool - plot some of the metrics or not
    :param epoch: int - epoch id
    :param max_sample: int - maximum number of points to try to plot for the xy plot large values cause memory issues
    :param plot_summary: bool - plot the metrics only as a combined summary image
    :return: dict
    """

    log = logging.getLogger(__name__)

    uncertainity_metrics = uct.metrics.get_all_metrics(predictions,
                                                       standard_deviation,
                                                       true_value,
                                                       verbose=verbose,
                                                       scaled=False,
                                                       num_bins=number_of_bins)

    log.debug(list(uncertainity_metrics.keys()))
    for_pd_uncertainity_metrics = {}
    for k, v in uncertainity_metrics.items():
        for subk, subv in v.items():
            if isinstance(subv, dict):
                for subsubk, subsubv in subv.items():
                    for_pd_uncertainity_metrics[subk + "_" + subsubk] = subsubv
            else:
                for_pd_uncertainity_metrics[subk] = subv

    log.debug(for_pd_uncertainity_metrics)

    df_tmp = pd.DataFrame(for_pd_uncertainity_metrics)

    if epoch is not None:
        fjname = "uncertainity_metrics_epoch_{}.json".format(epoch)
    else:
        fjname = "uncertainity_metrics.json"
    df_tmp.to_json(fjname)
    df_tmp.to_csv(fjname.split(".")[0] + ".csv", index=False)

    if plot is True:
        log.info("Epoch {} plot is {} summary plot is {}".format(epoch, plot, plot_summary))
        
        if x is None:
            x = np.array([ith for ith in range(len(predictions))])
        
        if len(predictions) > max_sample:
            n_sample = max_sample
            log.info("Max sample for xy plot will be {}".format(n_sample))
        else:
            n_sample = None
            log.info("Using all data points for xy plot")

        if plot_summary is False:
            log.info("Full plotting for epoch {}".format(epoch))
            plot_all(predictions, standard_deviation, true_value, x, epoch, n_sample, uncertainity_metrics)
        else:
            log.info("Summary plott on epoch {}".format(epoch))
            summary_plotting(predictions, standard_deviation, true_value, x, epoch, n_sample, uncertainity_metrics)

    return uncertainity_metrics

def find_fonts() -> Union[None, str]:
    """
    Function to find font files on a system for adding labels to images
    idea from https://stackoverflow.com/questions/15365405/python-finding-ttf-files
    :return: str
    """
    log = logging.getLogger(__name__)

    system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_path in system_fonts:
        if re.search(r"[a-zA-Z/]+/Arial[ a-zA-z]+.ttf$", font_path):
            return font_path

    log.info("No font file found using a default font")
    return None

def combine_images(image_names: list =(None),
                   images_per_row: int =3,
                   titles: list = None,
                   font: str = None,
                   fontsize: int = 40,
                   epoch: str = None,
                   dbug: bool = False
                   ) -> None:
    """
    Function to concatenate images on metrics. The following was consulted to write this function
    https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    :param image_names: tuple - file names of images to combine
    :param images_per_row: int - number of images per row
    :param titles: tuple - names to identify the file plots
    :param font: str - path to ttf font font file
    :param fontsize: int - font size
    :param epoch: str - filename unique parameter
    :return: None
    """

    log = logging.getLogger(__name__)

    log.info("Combining {}".format(" ".join(image_names)))

    x = len(image_names) % images_per_row
    if x != 0:
        log.info("Number of images {} and the number of images per row {} are not devisable without a remainder"
                 .format(len(image_names), images_per_row))

        while x != 0:
            images_per_row = images_per_row + 1
            x = len(image_names) % images_per_row
        log.info("Will set the number of images per row to {}".format(images_per_row))

    # ROWS
    i = 0
    rows = []
    n_row = int(len(image_names) / images_per_row)
    log.info("Expecting {} image rows".format(n_row))
    for c in range(n_row):

        j = (c + 1) * images_per_row

        row_images = image_names[i:j]
        log.debug("i {} j {}".format(i, j))
        log.debug("{} {}".format(row_images, len(row_images)))

        images = [Image.open(x) for x in row_images]
        widths = [i.size[0] for i in images]
        heights = [i.size[1] for i in images]

        new_width = sum(widths)
        new_height = max(heights)

        if titles is not None:
            labels = titles[i:j]

            if font is None:
                log.warning("using default font and fontsize")
                im_font = ImageFont.load_default()
            else:
                try:
                    im_font = ImageFont.truetype(font, fontsize)
                except OSError:
                    log.warning("Could no load font using default and default size")
                    im_font = ImageFont.load_default()

            if len(images) != len(labels):
                log.warning("Number of input files {} and titles {} must be the same".format(len(images), len(labels)))
                raise RuntimeError

            title_offset = 5
            new_im = Image.new('RGB', (new_width + (title_offset * len(labels)), new_height + 100), (255, 255, 255))
            x_offset = 0
            for ith, im in enumerate(images):
                log.debug("Title: {}".format(labels[ith]))
                new_im.paste(im, (x_offset + title_offset, 100))
                edit_new_im = ImageDraw.Draw(new_im)
                edit_new_im.text((x_offset + (int(title_offset / 2)), 0), "{}".format(labels[ith]), (0, 0, 0), im_font)
                x_offset = x_offset + im.size[0] + title_offset
        else:
            new_im = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            x_offset = 0
            for ith, im in enumerate(images):
                log.debug("Concatenating image {} to main image".format(ith))
                new_im.paste(im, (x_offset, 0))
                x_offset = x_offset + im.size[0]

        if dbug is True:
            new_im.save("row_{}.png".format(c))

        rows.append(new_im)
        i = j

    # COLUMNS
    widths = [i.size[0] for i in rows]
    heights = [i.size[1] for i in rows]

    new_width = max(widths)
    new_height = sum(heights)
    combined_im = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    x_offset = None

    y_offset = 0
    if x_offset is None:
        x_offset = [0] * len(rows)
    for ith, im in enumerate(rows):
        log.debug("Concatenating image {} to main image".format(ith))
        combined_im.paste(im, (x_offset[ith], y_offset))
        y_offset = y_offset + im.size[1]

    if epoch is not None:
        combined_im.save("uncertainity_metric_plots_epoch_{}.png".format(epoch))
        log.info("Combined plot saved as 'uncertainity_metric_plots_epoch_{}.png'".format(epoch))
    else:
        combined_im.save("uncertainity_metric_plots.png")
        log.info("Combined plot saved as uncertainity_metric_plots.png")

def how_many_models(model: torchani.models = None):
    """
    Function to find out how many ani models there are from a built in model or an ensemble
    :param model: torchani.models - ANI model network instance
    :return: int, None/int - number of model, if there is an ensemble class its index
    """

    log = logging.getLogger(__name__)

    models_found = 0
    ensemble_index = None
    for ith in range(0, len(model)):
        log.debug(f"Model[{ith}] type: {str(type(model[ith]))}")
        if isinstance(model[ith], torchani.models.BuiltinModel):
            log.info("Found BuiltinModel counting")
            models_found = models_found + 1
            log.info(f"Found {models_found} built in models.")
        if isinstance(model[ith], torchani.nn.Ensemble):
            log.info(f"Found Ensemble indexed {ith}")
            ensemble_index = ith
            models_found = len(model[ith])
            log.info(f"The ensemble has {models_found} models.")
            break

    return models_found, ensemble_index

def get_ani_parameter_distrbutions(model: torchani.models = None,
                                   species_order: tuple =("H", "C", "N", "O"),
                                   check: bool = True,
                                   prepend_key: str = ""):
    """
    Function to get the ANI (ANI1x by default) parameter distrbutions (weights and bias means and standard deviations)
    from the model and make them usable as initialization input.
    :param model: torchani.models - ANI model
    :param species_order: tuple - Element species order as a key mapping
    :param check: bool - Whether to check that the values are as expected for rho weights
    :return: dict - dictionary of dictionaries with keys for the layers and the the inner dict for the values
    """

    log = logging.getLogger(__name__)

    key_map = {k : str(i) for i, k in enumerate(species_order)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = torchani.models.ANI1x(periodic_table_index=True).to(device)
        n_models = len(model)
    else:
        n_models, ensemble_index = how_many_models(model)

    mean_weights, mean_biases = get_means(model, ensemble_length=n_models)
    std_weights, std_biases = get_std(model, ensemble_length=n_models)

    params = {}

    log.debug("Species key map: {}".format(key_map))

    for ik, k in enumerate(sorted(mean_weights.keys())):

        if ik == 0:
            log.debug(f"key: {k}")

        new_key = ".".join(k.split(".")[1:-1])
        for key, r in key_map.items():
            new_key = new_key.replace(key, r)
        if prepend_key != "":
            new_key = prepend_key + new_key

        log.debug(f"Old k {k} new key {new_key}")
        params[new_key] = {}
        params[new_key]["weight_mean"] = mean_weights[k]
        params[new_key]["weight_std"] = std_weights[k]
        params[new_key]["weight_rho"] = get_rho_from_sigma(std_weights[k])
        params[new_key]["bias_mean"] = mean_biases[k.replace("weight", "bias")]
        params[new_key]["bias_std"] = std_biases[k.replace("weight", "bias")]
        params[new_key]["bias_rho"] = get_rho_from_sigma(std_biases[k.replace("weight", "bias")])


    if check is True:
        log.info("The following comparison should provide the same numbers. "
                 "The RHO values should become the SIGMA through a soft plus function."
                 "https://arxiv.org/pdf/1906.05323v1.pdf. Here we check for one example that is true.")

        if round(std_weights['neural_networks.C.0.weight'][0][0].item(), 4) == round(
                torch.log(1 + torch.exp(params[prepend_key + '{}.0'.format(key_map["C"])]['weight_rho'][0][0])).item(),
                4):

            log.info("The results match as expected")
            log.info("{} == {}".format(round(std_weights['neural_networks.C.0.weight'][0][0].item(), 4),
                                       round(torch.log(1 + torch.exp(params[prepend_key + '{}.0'.format(key_map["C"])]
                                                                     ['weight_rho'][0][0])).item(), 4)
                                       )
                     )
        else:
            log.warning("The results don't match please check carefully")

    return params

def summary_plotting(predictions: np.ndarray,
                     standard_deviation: np.ndarray,
                     true_value: np.ndarray,
                     x: np.ndarray,
                     epoch: int,
                     n_sample: Union[int, None],
                     uncertainity_metrics: dict
                     ) -> None:
    """
    plot uncertainty metrics as individual plots and as a combined plot
    :param predictions: np.ndarray - iterable array of predicted values
    :param standard_deviation: np.ndarray - iterable array of standard deviations over the predictions
    :param true_value: np.ndarray - iterable array of true values
    :param x: np.ndarray - inputs for the predictions is None it will be a zero based index
    :param epoch: int - epoch id
    :param n_sample: int - maximum number of points to try to plot for the xy plot large values cause memory issues
    :param uncertainity_metrics: dict - metrics as dict for uncertainty from uncertainty tools
    :return:
    """

    log = logging.getLogger(__name__)

    fig, ax = plt.subplots(5, 3, figsize=(35, 35))

    # define limits for the plots
    y_origin = min(predictions - (2 * standard_deviation))
    y_origin = y_origin - (0.1 * y_origin)

    y_top = max(predictions + (2 * standard_deviation))
    y_top = y_top + (0.1 * y_top)

    # Plot xy
    # This plot can have overflow issues n_sample is used to prevent the overflow
    uct.plot_xy(predictions, standard_deviation, true_value, x, leg_loc=4, n_subset=n_sample, ax=ax[0, 0])
    ax[0, 0].legend(prop=dict(size=20))
    ax[0, 0].xaxis.label.set_size(25)
    ax[0, 0].yaxis.label.set_size(25)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=20)
    title = ax[0, 0].get_title()
    ax[0, 0].set_title(title, fontdict={"fontsize" : 25})


    # Plot intervals
    uct.plot_intervals(predictions, standard_deviation, true_value, ax=ax[0, 1], ylims=(y_origin, y_top),
                       num_stds_confidence_bound=2)
    ax[0, 1].legend(prop=dict(size=20))
    ax[0, 1].xaxis.label.set_size(25)
    ax[0, 1].yaxis.label.set_size(25)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=20)
    title = ax[0, 1].get_title()
    ax[0, 1].set_title(title, fontdict={"fontsize": 25})

    # Plot intervals_ordered
    uct.plot_intervals_ordered(predictions, standard_deviation, true_value, ax=ax[0, 2], ylims=(y_origin, y_top),
                               num_stds_confidence_bound=2)
    ax[0, 2].legend(prop=dict(size=20))
    ax[0, 2].xaxis.label.set_size(25)
    ax[0, 2].yaxis.label.set_size(25)
    ax[0, 2].tick_params(axis='both', which='major', labelsize=20)
    title = ax[0, 2].get_title()
    ax[0, 2].set_title(title, fontdict={"fontsize": 25})

    # Plot calibration
    uct.plot_calibration(predictions, standard_deviation, true_value, ax=ax[1, 0])
    ax[1, 0].legend(prop=dict(size=20))
    ax[1, 0].xaxis.label.set_size(25)
    ax[1, 0].yaxis.label.set_size(25)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=20)
    title = ax[1, 0].get_title()
    ax[1, 0].set_title(title, fontdict={"fontsize": 25})

    # Plot adversarial group calibration
    uct.plot_adversarial_group_calibration(predictions, standard_deviation, true_value, ax=ax[1, 1])
    ax[1, 1].legend(prop=dict(size=20))
    ax[1, 1].xaxis.label.set_size(25)
    ax[1, 1].yaxis.label.set_size(25)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=20)
    title = ax[1, 1].get_title()
    ax[1, 1].set_title(title, fontdict={"fontsize": 25})

    # Plot sharpness
    uct.plot_sharpness(standard_deviation, ax=ax[1, 2])
    ax[1, 2].legend(prop=dict(size=20))
    ax[1, 2].xaxis.label.set_size(25)
    ax[1, 2].yaxis.label.set_size(25)
    ax[1, 2].tick_params(axis='both', which='major', labelsize=20)
    title = ax[1, 2].get_title()
    ax[1, 2].set_title(title, fontdict={"fontsize": 25})

    # Plot residuals vs stds
    uct.plot_residuals_vs_stds(predictions, standard_deviation, true_value, ax=ax[2, 0])
    ax[2, 0].legend(prop=dict(size=20))
    ax[2, 0].xaxis.label.set_size(25)
    ax[2, 0].yaxis.label.set_size(25)
    ax[2, 0].tick_params(axis='both', which='major', labelsize=20)
    title = ax[2, 0].get_title()
    ax[2, 0].set_title(title, fontdict={"fontsize": 25})

    # parity plot with y error bars showing the uncertainty
    ax[2, 1].errorbar(true_value,
                predictions,
                yerr=standard_deviation,
                fmt="bo",
                ecolor="g",
                elinewidth=2.5,
                capsize=1.5,
                capthick=1.5,
                label="True Vs predicted energies (Ha) with uncert RMSE: {:.2f} Ha {:.2f} kcal/mol".format(
                    uncertainity_metrics["accuracy"]["rmse"],
                    hartree2kcalmol(uncertainity_metrics["accuracy"]["rmse"])),
                alpha=0.5
                )
    ax[2, 1].legend(fontsize=20)
    ax[2, 1].set_title("True energies against predicted energies", fontsize=25)
    ax[2, 1].set_xlabel("True Energies (Ha)", fontsize=25)
    ax[2, 1].set_ylabel("Predicted Energies and uncert (Ha)", fontsize=25)
    ax[2, 1].tick_params(axis='both', which='major', labelsize=20)
    ax[2, 1].grid(True)

    # Uncertainty against standard deviation
    error = np.array([abs(t - p) for t, p in zip(true_value, predictions)])

    sorted_std_inds = np.argsort(standard_deviation)
    sorted_std = standard_deviation[sorted_std_inds]
    sorted_error = error[sorted_std_inds]

    n_samples = len(standard_deviation)
    p20_ind = int(n_samples * 0.2)
    p50_ind = int(n_samples * 0.5)
    p80_ind = int(n_samples * 0.8)

    p20_thresh = sorted_std[p20_ind]
    p50_thresh = sorted_std[p50_ind]
    p80_thresh = sorted_std[p80_ind]

    p20_e_mean = hartree2kcalmol(np.mean(sorted_error[:p20_ind+1]))
    p20_e_max = hartree2kcalmol(np.max(sorted_error[:p20_ind+1]))

    p50_e_mean = hartree2kcalmol(np.mean(sorted_error[:p50_ind+1]))
    p50_e_max = hartree2kcalmol(np.max(sorted_error[:p50_ind+1]))

    p80_e_mean = hartree2kcalmol(np.mean(sorted_error[:p80_ind+1]))
    p80_e_max = hartree2kcalmol(np.max(sorted_error[:p80_ind+1]))

    ax[2, 2].plot(standard_deviation,
                  error,
                  "bo",
                  label="Standard deviation against error",
                  alpha=0.5)
    ax[2, 2].axvline(p20_thresh, linestyle='--', label=f'{p20_e_mean:.2f} ({p20_e_max:.2f}) kcal/mol mean (max) error - 20% data')
    ax[2, 2].axvline(p50_thresh, linestyle='--', label=f'{p50_e_mean:.2f} ({p50_e_max:.2f}) kcal/mol mean (max) error - 50% data')
    ax[2, 2].axvline(p80_thresh, linestyle='--', label=f'{p80_e_mean:.2f} ({p80_e_max:.2f}) kcal/mol mean (max) error - 80% data')
    ax[2, 2].legend(fontsize=20)
    ax[2, 2].set_title("Predicted energy uncertainty against absolute error in energy", fontsize=25)
    ax[2, 2].set_xlabel("Uncertainty (Ha)", fontsize=25)
    ax[2, 2].set_ylabel("Error (Ha)", fontsize=25)
    ax[2, 2].tick_params(axis='both', which='major', labelsize=20)
    ax[2, 2].grid(True)

    # CDF/PDF absolute errors
    # https://stackoverflow.com/questions/25577352/plotting-cdf-of-a-pandas-series-in-python
    # Use group by uncase the valu is not a one off whilst this is unlikely here it is possible
    ae_df = pd.DataFrame(pd.Series([abs(t - p) for t, p in zip(true_value, predictions)], name="abs_errs"))
    err_df = ae_df.groupby("abs_errs")["abs_errs"].agg("count").pipe(pd.DataFrame).rename(columns={"abs_errs": "freq"})

    err_df["PDF"] = err_df["freq"] / sum(err_df["freq"])

    err_df["CDF"] = err_df["PDF"].cumsum()
    err_df = err_df.reset_index()
    err_df.to_csv("abs_errors_epoch_{}.csv".format(epoch), index=False)

    err_df.plot(x="abs_errs", y=["PDF", "CDF"], grid=True, ax=ax[3, 0], colormap="rainbow", fontsize=20)
    ax[3, 0].set_xlabel("Absolute Error", fontsize=25)
    ax[3, 0].set_ylabel("Distribution", fontsize=25)
    ax[3, 0].set_title("Distribution Functions for Absolute Errors", fontsize=25)
    ax[3, 0].legend(prop=dict(size=20))

    # CDF/PDF errors
    e_df = pd.DataFrame(pd.Series([t - p for t, p in zip(true_value, predictions)], name="errs"))
    err_df = e_df.groupby("errs")["errs"].agg("count").pipe(pd.DataFrame).rename(columns={"errs": "freq"})

    err_df["PDF"] = err_df["freq"] / sum(err_df["freq"])

    err_df["CDF"] = err_df["PDF"].cumsum()
    err_df = err_df.reset_index()
    err_df.to_csv("errors_epoch_{}.csv".format(epoch), index=False)

    err_df.plot(x="errs", y=["PDF", "CDF"], grid=True, ax=ax[3, 1], colormap="rainbow", fontsize=20)
    ax[3, 1].set_xlabel("Signed Errors (True - Predicted)", fontsize=25)
    ax[3, 1].set_ylabel("Distribution", fontsize=25)
    ax[3, 1].set_title("Distribution Functions for Signed Errors", fontsize=25)
    ax[3, 1].legend(prop=dict(size=20))

    # CDF/PDF uncertainty
    sigma_df = pd.DataFrame(pd.Series(standard_deviation, name="sd"))
    sd_df = sigma_df.groupby("sd")["sd"].agg("count").pipe(pd.DataFrame).rename(columns={"sd": "freq"})

    sd_df["PDF"] = sd_df["freq"] / sum(sd_df["freq"])

    sd_df["CDF"] = sd_df["PDF"].cumsum()
    sd_df = sd_df.reset_index()
    sd_df.to_csv("standard_deviation_epoch_{}.csv".format(epoch), index=False)

    sd_df.plot(x="sd", y=["PDF", "CDF"], grid=True, ax=ax[3, 2], colormap="rainbow", fontsize=20)
    ax[3, 2].set_xlabel("Standard Deviations", fontsize=25)
    ax[3, 2].set_ylabel("Distrbution", fontsize=25)
    ax[3, 2].set_title("Distribution Functions for Standard Deviations", fontsize=25)
    ax[3, 2].legend(prop=dict(size=20))

    # CDF/PDF uncertainty with prediction points overlayed
    ax2 = ax[4, 0].twinx()
    leg = ax[4, 0].plot(standard_deviation, [abs(t - p) for t, p in zip(true_value, predictions)], "bo",
                  label="Absolute Errors (Ha)", alpha=0.5)
    log.debug(leg)
    lab = [leg[0].get_label()]
    log.debug(lab)
    ax[4, 0].set_xlabel("Standard Deviations", fontsize=25)
    ax[4, 0].set_ylabel("Absolute error (Ha)", fontsize=25)
    ax[4, 0].set_title("Standard Deviations Against PDF, CDF and Absolute Errors", fontsize=25)

    # ax2
    sd_df.plot(x="sd", y=["PDF", "CDF"], grid=True, ax=ax2, colormap="rainbow", fontsize=20)
    legs, labs = ax2.get_legend_handles_labels()
    log.debug(legs)
    log.debug(labs)
    legs = legs + leg
    labs = labs + lab
    log.debug(legs)
    log.debug(labs)
    ax2.set_ylabel("Distrbution", fontsize=25)
    ax2.legend(legs, labs, prop=dict(size=20))

    # Parity plot coloured by the standard deviation
    sc = ax[4, 1].scatter(true_value,
                          predictions,
                          c=standard_deviation,
                          cmap="rainbow",
                          vmin=min(standard_deviation),
                          vmax=max(standard_deviation),
                          alpha=0.5,

                          )
    plt.colorbar(sc, ax=ax[4, 1])
    ax[4, 1].legend(fontsize=20)
    ax[4, 1].set_title("True energies against predicted energies coloured by uncertainty", fontsize=25)
    ax[4, 1].set_xlabel("True Energies (Ha)", fontsize=25)
    ax[4, 1].set_ylabel("Predicted Energies and uncert (Ha)", fontsize=25)
    ax[4, 1].tick_params(axis='both', which='major', labelsize=20)
    ax[4, 1].grid(True)

    # Parity plot coloured by the abs error
    abs_errors = [abs(tr - pr) for tr, pr in zip(true_value, predictions)]
    sc = ax[4, 2].scatter(true_value,
                          predictions,
                          c=abs_errors,
                          cmap="rainbow",
                          vmin=min(abs_errors),
                          vmax=max(abs_errors),
                          alpha=0.5
                          )
    plt.colorbar(sc, ax=ax[4, 2])
    ax[4, 2].legend(fontsize=20)
    ax[4, 2].set_title("True energies against predicted energies coloured by abs error (Ha)", fontsize=25)
    ax[4, 2].set_xlabel("True Energies (Ha)", fontsize=25)
    ax[4, 2].set_ylabel("Predicted Energies (Ha)", fontsize=25)
    ax[4, 2].tick_params(axis='both', which='major', labelsize=20)
    ax[4, 2].grid(True)

    plt.tight_layout()

    try:
        plt.savefig("uncertainity_metrics_epoch_{}.png".format(epoch))
        log.info(f"saving plot to uncertainity_metrics_epoch_{epoch}.png")
        plt.close()
    except OverflowError as oerr:
        plt.close()
        log.info("Overflow error encountered no summary plot for epoch {}".format(epoch))

def plot_all(predictions: np.ndarray,
             standard_deviation: np.ndarray,
             true_value: np.ndarray,
             x: np.ndarray,
             epoch: int,
             n_sample: Union[int, None],
             uncertainity_metrics: dict
             ) -> None:
    """
    plot uncertainty metrics as individual plots and as a combined plot
    :param predictions: np.ndarray - iterable array of predicted values
    :param standard_deviation: np.ndarray - iterable array of standard deviations over the predictions
    :param true_value: np.ndarray - iterable array of true values
    :param x: np.ndarray - inputs for the predictions is None it will be a zero based index
    :param epoch: int - epoch id
    :param n_sample: int - maximum number of points to try to plot for the xy plot large values cause memory issues
    :param uncertainity_metrics: dict - metrics for uncertainty from uncertainty tools
    :return:
    """

    log = logging.getLogger(__name__)

    images_names = []
    # Plot xy
    # This plot can have overflow issues n_sample is used to prevent the overflow
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_xy(predictions, standard_deviation, true_value, x, leg_loc=4, n_subset=n_sample, ax=ax)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()

    try:
        plt.savefig("uncertainity_metric_xy_plot_epoch_{}_sub_sample_{}.png".format(epoch, n_sample))
        images_names.append("uncertainity_metric_xy_plot_epoch_{}_sub_sample_{}.png".format(epoch, n_sample))
    except OverflowError as oerr:
        log.info("Skipping xy plot as it got the following err {}".format(oerr))
    plt.close()

    # Plot intervals
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_intervals(predictions, standard_deviation, true_value, ax=ax)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    try:
        plt.savefig("uncertainity_metric_intervals_plot_epoch_{}.png".format(epoch))
        images_names.append("uncertainity_metric_intervals_plot_epoch_{}.png".format(epoch))
    except OverflowError as oerr:
        log.info("Skipping interval plot as it got the following err {}".format(oerr))
    plt.close()

    # Plot intervals_ordered
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_intervals_ordered(predictions, standard_deviation, true_value, ax=ax)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    plt.savefig("uncertainity_metric_ordered_intervals_plot_epoch_{}.png".format(epoch))
    plt.close()
    images_names.append("uncertainity_metric_ordered_intervals_plot_epoch_{}.png".format(epoch))

    # Plot calibration
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_calibration(predictions, standard_deviation, true_value, ax=ax)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    try:
        plt.savefig("uncertainity_metric_calibration_plot_epoch_{}.png".format(epoch))
        images_names.append("uncertainity_metric_calibration_plot_epoch_{}.png".format(epoch))
    except OverflowError as oerr:
        log.info("Skipping calibration plot as it got the following err {}".format(oerr))
    plt.close()

    # Plot adversarial group calibration
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_adversarial_group_calibration(predictions, standard_deviation, true_value, ax=ax)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    try:
        plt.savefig("uncertainity_metric_adversarial_plot_epoch_{}.png".format(epoch))
        images_names.append("uncertainity_metric_adversarial_plot_epoch_{}.png".format(epoch))
    except OverflowError as oerr:
        log.info("Skipping adverserial plot as it got the following err {}".format(oerr))
    plt.close()

    # Plot sharpness
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_sharpness(standard_deviation, ax=ax)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    try:
        plt.savefig("uncertainity_metric_sharpness_plot_epoch_{}.png".format(epoch))
        images_names.append("uncertainity_metric_sharpness_plot_epoch_{}.png".format(epoch))
    except OverflowError as oerr:
        log.info("Skipping sharpness plot as it got the following err {}".format(oerr))
    plt.close()

    # Plot residuals vs stds
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    uct.plot_residuals_vs_stds(predictions, standard_deviation, true_value, ax=ax)
    plt.tight_layout()
    try:
        plt.savefig("uncertainity_metric_residuals_against_sigma_plot_epoch_{}.png".format(epoch))
        images_names.append("uncertainity_metric_residuals_against_sigma_plot_epoch_{}.png".format(epoch))
    except OverflowError as oerr:
        log.info("Skipping residuals against sigma plot as it got the following err {}".format(oerr))

    plt.close()

    # parity plot with y error bars showing the uncertainty
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.errorbar(true_value,
                predictions,
                yerr=standard_deviation,
                fmt="bo",
                ecolor="g",
                elinewidth=2.5,
                capsize=1.5,
                capthick=1.5,
                label="True Vs predicted energies (Ha) with uncert RMSE: {:.2f} Ha {:.2f} kcal/mol".format(
                    uncertainity_metrics["accuracy"]["rmse"],
                    hartree2kcalmol(uncertainity_metrics["accuracy"]["rmse"])),
                alpha=0.5
                )
    ax.legend()
    ax.set_title("True energies against predicted energies")
    ax.set_xlabel("True Energies (Ha)")
    ax.set_ylabel("Predicted Energies and uncert (Ha)")
    ax.grid(True)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    plt.savefig("uncertainity_metric_parity_plot_epoch_{}.png".format(epoch))
    plt.close()
    images_names.append("uncertainity_metric_parity_plot_epoch_{}.png".format(epoch))

    # Uncertainty against standard deviation
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.plot(standard_deviation,
            [abs(t - p) for t, p in zip(true_value, predictions)],
            "bo",
            label="Standard deviation against error",
            alpha=0.5)
    ax.legend()
    ax.set_title("Predicted energy uncertainty against absolute error in energy")
    ax.set_xlabel("Uncertainty (Ha)")
    ax.set_ylabel("Error (Ha)")
    ax.grid(True)
    ax.legend(prop=dict(size=20))
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    title = ax.get_title()
    ax.set_title(title, fontdict={"fontsize": 25})
    plt.tight_layout()
    plt.savefig("uncertainity_metric_uparity_plot_epoch_{}.png".format(epoch))
    plt.close()
    images_names.append("uncertainity_metric_uparity_plot_epoch_{}.png".format(epoch))

    # CDF/PDF absolute errors
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ae_df = pd.DataFrame(pd.Series([abs(t - p) for t, p in zip(true_value, predictions)], name="abs_errs"))
    err_df = ae_df.groupby("abs_errs")["abs_errs"].agg("count").pipe(pd.DataFrame).rename(columns={"abs_errs": "freq"})

    err_df["PDF"] = err_df["freq"] / sum(err_df["freq"])

    err_df["CDF"] = err_df["PDF"].cumsum()
    err_df = err_df.reset_index()
    err_df.to_csv("abs_errors_epoch_{}.csv".format(epoch), index=False)

    err_df.plot(x="abs_errs", y=["PDF", "CDF"], grid=True, ax=ax, colormap="rainbow", fontsize=20)
    ax.set_xlabel("Absolute Error", fontsize=25)
    ax.set_ylabel("Distribution", fontsize=25)
    ax.set_title("Distribution Functions for Absolute Errors", fontsize=25)
    ax.legend(prop=dict(size=20))
    plt.tight_layout()
    plt.savefig("pdf_cdf_absolute_error_epoch_{}.png".format(epoch))
    plt.close()
    images_names.append("pdf_cdf_absolute_error_epoch_{}.png".format(epoch))

    # CDF/PDF errors
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    e_df = pd.DataFrame(pd.Series([t - p for t, p in zip(true_value, predictions)], name="errs"))
    err_df = e_df.groupby("errs")["errs"].agg("count").pipe(pd.DataFrame).rename(columns={"errs": "freq"})

    err_df["PDF"] = err_df["freq"] / sum(err_df["freq"])

    err_df["CDF"] = err_df["PDF"].cumsum()
    err_df = err_df.reset_index()
    err_df.to_csv("errors_epoch_{}.csv".format(epoch), index=False)

    err_df.plot(x="errs", y=["PDF", "CDF"], grid=True, ax=ax, colormap="rainbow", fontsize=20)
    ax.set_xlabel("Signed Errors (True - Predicted)", fontsize=25)
    ax.set_ylabel("Distribution", fontsize=25)
    ax.set_title("Distribution Functions for Signed Errors", fontsize=25)
    ax.legend(prop=dict(size=20))
    plt.tight_layout()
    plt.savefig("pdf_cdf_signed_error_epoch_{}.png".format(epoch))
    plt.close()
    images_names.append("pdf_cdf_signed_error_epoch_{}.png".format(epoch))

    # CDF/PDF uncertainty with prediction points overlayed
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    sigma_df = pd.DataFrame(pd.Series(standard_deviation, name="sd"))
    sd_df = sigma_df.groupby("sd")["sd"].agg("count").pipe(pd.DataFrame).rename(columns={"sd": "freq"})

    sd_df["PDF"] = sd_df["freq"] / sum(sd_df["freq"])

    sd_df["CDF"] = sd_df["PDF"].cumsum()
    sd_df = sd_df.reset_index()

    sd_df.plot(x="sd", y=["PDF", "CDF"], grid=True, ax=ax, colormap="rainbow", fontsize=20)
    ax.set_xlabel("Standard Deviations", fontsize=25)
    ax.set_ylabel("Distrbution", fontsize=25)
    ax.set_title("Distribution Functions for Standard Deviations", fontsize=25)
    ax.legend(prop=dict(size=20))
    plt.tight_layout()
    plt.savefig("pdf_cdf_uncertainty_epoch_{}.png".format(epoch))
    plt.close()
    images_names.append("pdf_cdf_uncertainty_epoch_{}.png".format(epoch))


    # CDF/PDF uncertainty with prediction points overlayed
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax2 = ax.twinx()

    # ax
    leg = ax.plot(standard_deviation, [abs(t - p) for t, p in zip(true_value, predictions)], "bo",
             label="Absolute Errors (Ha)", alpha=0.5)
    log.info(leg)
    lab = [leg[0].get_label()]
    log.info(lab)
    ax.set_xlabel("Standard Deviations", fontsize=25)
    ax.set_ylabel("Absolute error (Ha)", fontsize=25)
    ax.set_title("Standard Deviations Against PDF, CDF and Absolute Errors", fontsize=25)

    # ax2
    sd_df.plot(x="sd", y=["PDF", "CDF"], grid=True, ax=ax2, colormap="rainbow", fontsize=20)
    legs, labs = ax2.get_legend_handles_labels()
    log.info(legs)
    log.info(labs)
    legs = legs + leg
    labs = labs + lab
    log.info(legs)
    log.info(labs)
    ax2.set_ylabel("Distrbution", fontsize=25)
    ax2.legend(legs, labs, prop=dict(size=20))
    plt.tight_layout()
    plt.savefig("uncertainty_against_abs_errors_pdf_cdf_epoch_{}.png".format(epoch))
    plt.close()

    font = find_fonts()
    combine_images(image_names=images_names,
                   images_per_row=3,
                   titles=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
                   epoch=epoch,
                   font=font,
                   fontsize=40
                   )

def _update_bnn_variance_with_tests(model,
                       bnn_prior: dict,
                       params_prior: dict,
                       bayesian_depth: int = None,
                       set_prior_explicitly: bool = False,
                       set_rho_explicitly: bool = False,
                       prior_key=""):
    """
    Function to convert ani to BNN to a set depth on the network
    :param model : torch.nn.Sequential - pytorch network
    :param bnn_prior: dict - output of chosen prior from bnn_priors
    :param params_prior : dict - dictionary keys are layers named by numbers as in model.named_parameters()
                                 and to sub-dictionary where keys of weight_mean, weight_std, bias_mean and bias_std
                                 set prior
    :param bayesian_depth : int - depth of layers to convert to bayesian
    :param set_explicitly : bool - set the rho values explicitly from user definition
    :param prior_key: str - SHOULD NOT BE SET EXPLICITLY TO ANYTHING OTHER THAN ''
                            IT IS USED IN THE RECURSIVE CALLS BY THIS FUNCTION
    return None - model is set to Bayesian layers inplace
    """

    log = logging.getLogger(__name__)

    log.debug("Prior key start of function: '{}'".format(prior_key))

    for name, value in list(model._modules.items()):

        if model._modules[name]._modules:
            log.debug("model._modules[name]._modules: True")
            log.debug("Name: {}".format(name))
            if prior_key == "":
                prior_key = name
                update_bnn_variance(model._modules[name],
                                   bnn_prior,
                                   params_prior,
                                   bayesian_depth=bayesian_depth,
                                   set_prior_explicitly=set_prior_explicitly,
                                   set_rho_explicitly=set_rho_explicitly,
                                   prior_key=prior_key)
                prior_key = ""
            else:
                prior_key_mid = prior_key + "." + name
                update_bnn_variance(model._modules[name],
                                   bnn_prior,
                                   params_prior,
                                   bayesian_depth=bayesian_depth,
                                   set_prior_explicitly=set_prior_explicitly,
                                   set_rho_explicitly=set_rho_explicitly,
                                   prior_key=prior_key_mid)

        elif "Conv" in model._modules[name].__class__.__name__:
            log.debug("'Conv' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            if bayesian_depth is None:
                # None = apply to all layers
                if bayesian_depth is None:

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in note if we pass them in they should be transformed as rho values from
                    # standard deviations.
                    if set_rho_explicitly is False:
                        log.info(f"Original rho values\n\n{model._modules[name].rho_weight.data}\n")

                        tmp_mus = torch.clone(model._modules[name].mu_weight.data)
                        expected_update = torch.log(torch.expm1(bnn_prior["moped_delta"] * torch.abs(tmp_mus)) + 1e-20)

                        model._modules[name].rho_weight.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_weight.data,
                                bnn_prior["moped_delta"]
                            )
                        )

                        updated_rhos = torch.clone(model._modules[name].rho_weight.data)
                        updated_rhos_np = updated_rhos.detach().cpu().numpy()

                        log.info(f"Should be different to the last ones\n\n{model._modules[name].rho_weight.data}\n"
                                 f"-----\n\n")

                        for ith, ent in enumerate(expected_update):
                            for jth, elt in enumerate(ent):
                                if elt == updated_rhos_np[ith][jth]:
                                    pass
                                else:
                                    log.info(
                                        f"Entries expected_update[{ith}][{jth}] != updated_rhos_np[{ith}][{jth}] : "
                                        f"{expected_update[ith][jth]} != {updated_rhos_np[ith][jth]}")

                        if all(elt == updated_rhos_np[ith][jth] for ith, ent in enumerate(expected_update)
                               for jth, elt in enumerate(ent)):
                            log.info("Appears the updates have the same effect")
                        else:
                            log.warning("WARNING - Appears the updates have varying effects")

                        if all(elt != updated_rhos_np[ith][jth] for ith, ent in enumerate(expected_update)
                               for jth, elt in enumerate(ent)):
                            log.info("Appears all updated Rho values are different to the original rho values")
                        else:
                            log.warning("WARNING - Appears the updates some elements with the same values")

                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        if set_rho_explicitly is False:
                            log.info(f"Original rho values\n\n{model._modules[name].rho_bias.data}\n")

                            tmp_mus = torch.clone(model._modules[name].mu_bias.data)
                            expected_update = torch.log(
                                torch.expm1(bnn_prior["moped_delta"] * torch.abs(tmp_mus)) + 1e-20)

                            model._modules[name].rho_bias.data.copy_(
                                dnn_to_bnn.get_rho(
                                    model._modules[name].mu_bias.data,
                                    bnn_prior["moped_delta"]
                                )
                            )

                            updated_rhos = torch.clone(model._modules[name].rho_bias.data)
                            updated_rhos_np = updated_rhos.detach().cpu().numpy()

                            log.info(f"Should be different to the last ones\n\n{model._modules[name].rho_bias.data}\n"
                                     f"-----\n\n")

                            for ith, ent in enumerate(expected_update):
                                try:
                                    for jth, elt in enumerate(ent):
                                        if elt == updated_rhos_np[ith][jth]:
                                            pass
                                        else:
                                            log.info(
                                                f"Entries expected_update[{ith}][{jth}] != updated_rhos_np[{ith}][{jth}] : "
                                                f"{expected_update[ith][jth]} != {updated_rhos_np[ith][jth]}")
                                except TypeError as terr:

                                    if ent == updated_rhos_np[ith]:
                                        pass
                                    else:
                                        log.info(
                                            f"Entries expected_update[{ith}] != updated_rhos_np[{ith}] : "
                                            f"{expected_update[ith]} != {updated_rhos_np[ith]}")

                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

                # int = apply to the layer specified by int and all subsequent layers
                elif int(name) >= bayesian_depth:

                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_weight.data,
                                bnn_prior["moped_delta"]
                            )
                        )
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(
                                dnn_to_bnn.get_rho(
                                    model._modules[name].mu_bias.data,
                                    bnn_prior["moped_delta"]
                                )
                            )

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

                else:
                    pass

        elif "Linear" in model._modules[name].__class__.__name__:
            log.info("'Linear' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            # None = apply to all layers
            if bayesian_depth is None:

                # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                # as the values we pass in note if we pass them in they should be transformed as rho values from
                # standard deviations.
                if set_rho_explicitly is False:
                    log.info(f"Original rho values\n\n{model._modules[name].rho_weight.data}\n")

                    tmp_mus = torch.clone(model._modules[name].mu_weight.data)
                    expected_update = torch.log(torch.expm1(bnn_prior["moped_delta"] * torch.abs(tmp_mus)) + 1e-20)

                    model._modules[name].rho_weight.data.copy_(
                        dnn_to_bnn.get_rho(
                            model._modules[name].mu_weight.data,
                            bnn_prior["moped_delta"]
                        )
                    )


                    updated_rhos = torch.clone(model._modules[name].rho_weight.data)
                    updated_rhos_np = updated_rhos.detach().cpu().numpy()

                    log.info(f"Should be different to the last ones\n\n{model._modules[name].rho_weight.data}\n"
                             f"-----\n\n")

                    for ith, ent in enumerate(expected_update):
                        for jth, elt in enumerate(ent):
                            if elt == updated_rhos_np[ith][jth]:
                                pass
                            else:
                                log.info(f"Entries expected_update[{ith}][{jth}] != updated_rhos_np[{ith}][{jth}] : "
                                         f"{expected_update[ith][jth]} != {updated_rhos_np[ith][jth]}")

                    if all(elt == updated_rhos_np[ith][jth] for ith, ent in enumerate(expected_update)
                           for jth, elt in enumerate(ent)):
                        log.info("Appears the updates have the same effect")
                    else:
                        log.warning("WARNING - Appears the updates have varying effects")

                    if all(elt != updated_rhos_np[ith][jth] for ith, ent in enumerate(expected_update)
                           for jth, elt in enumerate(ent)):
                        log.info("Appears all updated Rho values are different to the original rho values")
                    else:
                        log.warning("WARNING - Appears the updates some elements with the same values")

                elif set_rho_explicitly is True:
                    model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                # If there are biases used in the model set them to the values we have
                if model._modules[name].mu_bias is not None:

                    # 3. Set the mean of the bias distribution
                    if set_rho_explicitly is False:
                        log.info(f"Original rho values\n\n{model._modules[name].rho_bias.data}\n")

                        tmp_mus = torch.clone(model._modules[name].mu_bias.data)
                        expected_update = torch.log(torch.expm1(bnn_prior["moped_delta"] * torch.abs(tmp_mus)) + 1e-20)

                        model._modules[name].rho_bias.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_bias.data,
                                bnn_prior["moped_delta"]
                            )
                        )

                        updated_rhos = torch.clone(model._modules[name].rho_bias.data)
                        updated_rhos_np = updated_rhos.detach().cpu().numpy()

                        log.info(f"Should be different to the last ones\n\n{model._modules[name].rho_bias.data}\n"
                                 f"-----\n\n")

                        for ith, ent in enumerate(expected_update):
                            try:
                                for jth, elt in enumerate(ent):
                                    if elt == updated_rhos_np[ith][jth]:
                                        pass
                                    else:
                                        log.info(
                                            f"Entries expected_update[{ith}][{jth}] != updated_rhos_np[{ith}][{jth}] : "
                                            f"{expected_update[ith][jth]} != {updated_rhos_np[ith][jth]}")
                            except TypeError as terr:

                                if ent == updated_rhos_np[ith]:
                                    pass
                                else:
                                    log.info(
                                        f"Entries expected_update[{ith}] != updated_rhos_np[{ith}] : "
                                        f"{expected_update[ith]} != {updated_rhos_np[ith]}")

                    elif set_rho_explicitly is True:
                        model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            # int = apply to the layer specified by int and all subsequent layers
            elif int(name) >= bayesian_depth:

                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_weight.data,
                                bnn_prior["moped_delta"]
                            )
                        )
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(
                                dnn_to_bnn.get_rho(
                                    model._modules[name].mu_bias.data,
                                    bnn_prior["moped_delta"]
                                )
                            )

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            else:
                pass

        elif "LSTM" in model._modules[name].__class__.__name__:
            log.debug("'LSTM' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            # None = apply to all layers
            if bayesian_depth is None:

                # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                # as the values we pass in note if we pass them in they should be transformed as rho values from
                # standard deviations.
                if set_rho_explicitly is False:
                    log.info(f"Original rho values\n\n{model._modules[name].rho_weight.data}\n")

                    tmp_mus = torch.clone(model._modules[name].mu_weight.data)
                    expected_update = torch.log(torch.expm1(bnn_prior["moped_delta"] * torch.abs(tmp_mus)) + 1e-20)

                    model._modules[name].rho_weight.data.copy_(
                        dnn_to_bnn.get_rho(
                            model._modules[name].mu_weight.data,
                            bnn_prior["moped_delta"]
                        )
                    )

                    updated_rhos = torch.clone(model._modules[name].rho_weight.data)
                    updated_rhos_np = updated_rhos.detach().cpu().numpy()

                    log.info(f"Should be different to the last ones\n\n{model._modules[name].rho_weight.data}\n"
                             f"-----\n\n")

                    for ith, ent in enumerate(expected_update):
                        for jth, elt in enumerate(ent):
                            if elt == updated_rhos_np[ith][jth]:
                                pass
                            else:
                                log.info(f"Entries expected_update[{ith}][{jth}] != updated_rhos_np[{ith}][{jth}] : "
                                         f"{expected_update[ith][jth]} != {updated_rhos_np[ith][jth]}")

                    if all(elt == updated_rhos_np[ith][jth] for ith, ent in enumerate(expected_update)
                           for jth, elt in enumerate(ent)):
                        log.info("Appears the updates have the same effect")
                    else:
                        log.warning("WARNING - Appears the updates have varying effects")

                    if all(elt != updated_rhos_np[ith][jth] for ith, ent in enumerate(expected_update)
                           for jth, elt in enumerate(ent)):
                        log.info("Appears all updated Rho values are different to the original rho values")
                    else:
                        log.warning("WARNING - Appears the updates some elements with the same values")

                elif set_rho_explicitly is True:
                    model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                # If there are biases used in the model set them to the values we have
                if model._modules[name].mu_bias is not None:

                    # 3. Set the mean of the bias distribution
                    if set_rho_explicitly is False:
                        log.info(f"Original rho values\n\n{model._modules[name].rho_bias.data}\n")

                        tmp_mus = torch.clone(model._modules[name].mu_bias.data)
                        expected_update = torch.log(torch.expm1(bnn_prior["moped_delta"] * torch.abs(tmp_mus)) + 1e-20)

                        model._modules[name].rho_bias.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_bias.data,
                                bnn_prior["moped_delta"]
                            )
                        )

                        updated_rhos = torch.clone(model._modules[name].rho_bias.data)
                        updated_rhos_np = updated_rhos.detach().cpu().numpy()

                        log.info(f"Should be different to the last ones\n\n{model._modules[name].rho_bias.data}\n"
                                 f"-----\n\n")

                        for ith, ent in enumerate(expected_update):
                            try:
                                for jth, elt in enumerate(ent):
                                    if elt == updated_rhos_np[ith][jth]:
                                        pass
                                    else:
                                        log.info(
                                            f"Entries expected_update[{ith}][{jth}] != updated_rhos_np[{ith}][{jth}] : "
                                            f"{expected_update[ith][jth]} != {updated_rhos_np[ith][jth]}")
                            except TypeError as terr:

                                if ent == updated_rhos_np[ith]:
                                    pass
                                else:
                                    log.info(
                                        f"Entries expected_update[{ith}] != updated_rhos_np[{ith}] : "
                                        f"{expected_update[ith]} != {updated_rhos_np[ith]}")

                    elif set_rho_explicitly is True:
                        model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            # int = apply to the layer specified by int and all subsequent layers
            elif int(name) >= bayesian_depth:

                if set_rho_explicitly is False:
                    # NOTE get_rho multiplies the mean with delta to get the sigma value
                    model._modules[name].rho_weight.data.copy_(
                        dnn_to_bnn.get_rho(
                            model._modules[name].mu_weight.data,
                            bnn_prior["moped_delta"]
                        )
                    )
                elif set_rho_explicitly is True:
                    model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                # If there are biases used in the model set them to the values we have
                if model._modules[name].mu_bias is not None:

                    # 3. Set the mean of the bias distribution
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_bias.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_bias.data,
                                bnn_prior["moped_delta"]
                            )
                        )

                    # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            else:
                pass
        else:
            pass
