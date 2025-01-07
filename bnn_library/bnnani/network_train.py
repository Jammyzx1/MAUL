#!/usr/bin/env python

import math
import os
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard
import inspect
import torchani
import tqdm
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
# helper function to convert energy unit from Hartree to kcal/mol
from bayesian_torch.utils.util import predictive_entropy, mutual_information
from torchani.units import hartree2kcalmol
from bnnani import network_classes, network_test
from bnnani.network_utilities import get_uncertainity_estimation_metrics, get_uncertainity_metrics, log_likelihood
import logging

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

__title__ = os.path.basename(__file__)


def get_adamw_optimizer(h_network,
                        c_network,
                        n_network,
                        o_network,
                        learning_rate: float,
                        bayesian_depth: int,
                        use_ani_wd: bool = True,
                        bayesian: bool = False,
                        mu_opt: bool = True,
                        sig_opt: bool = True
                        ):
    """
    Function to get the appropiate AdamW optimizer for the chosen ANI network
    :param h_network: torch.nn.Sequential - Network architecture for Hydrogen
    :param c_network: torch.nn.Sequential - Network architecture for Carbon
    :param n_network: torch.nn.Sequential - Network architecture for Nitrogen
    :param o_network: torch.nn.Sequential - Network architecture for Oxygen
    :param learning_rate: float - initial learning rate
    :param bayesian_depth: None or int - depth of bayesian network
    :param use_ani_wd: bool - use ani weight decay or not
    :param bayesian: bool - is a bayesian network
    :param mu_opt: bool - flag specifying whether to train weight means
    :param sig_opt: bool - flag specifying whether to train weight sigmas
    :returns: torch.optim.AdamW - AdamW optimizer
    """
    log = logging.getLogger(__name__)

    if mu_opt:
        start_ind = 0
    else:
        start_ind = 1
    if sig_opt:
        end_ind = 2
    else:
        end_ind = 1

    if bayesian_depth is None and bayesian is True:
        log.info("Network is Bayesian for all layers. Attaching weights for mu and rho to AdamW optimizer")
        AdamW = torch.optim.AdamW(
            [
                # H networks
                {"params": [h_network[0].mu_weight, h_network[0].rho_weight][start_ind:end_ind]},
                {"params": [h_network[2].mu_weight, h_network[2].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.00001},
                {"params": [h_network[4].mu_weight, h_network[4].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.000001},
                {"params": [h_network[6].mu_weight, h_network[6].rho_weight][start_ind:end_ind]},
                # C networks
                {"params": [c_network[0].mu_weight, c_network[0].rho_weight][start_ind:end_ind]},
                {"params": [c_network[2].mu_weight, c_network[2].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.00001},
                {"params": [c_network[4].mu_weight, c_network[4].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.000001},
                {"params": [c_network[6].mu_weight, c_network[6].rho_weight][start_ind:end_ind]},
                # N networks
                {"params": [n_network[0].mu_weight, n_network[0].rho_weight][start_ind:end_ind]},
                {"params": [n_network[2].mu_weight, n_network[2].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.00001},
                {"params": [n_network[4].mu_weight, n_network[4].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.000001},
                {"params": [n_network[6].mu_weight, n_network[6].rho_weight][start_ind:end_ind]},
                # O networks
                {"params": [o_network[0].mu_weight, o_network[0].rho_weight][start_ind:end_ind]},
                {"params": [o_network[2].mu_weight, o_network[2].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.00001},
                {"params": [o_network[4].mu_weight, o_network[4].rho_weight][start_ind:end_ind],
                 "weight_decay": 0.000001},
                {"params": [o_network[6].mu_weight, o_network[6].rho_weight][start_ind:end_ind]},
            ],
            lr=learning_rate,
        )

    elif bayesian_depth is not None and bayesian is True:
        log.info("Some layers are Bayesian. Attaching weights for mu and rho on Bayesian layers to AdamW optimizer")
        optimzable = []

        # Assumption that you have [layer type, activation] * n hence only even
        # layers need updating
        if use_ani_wd is False:
            for ith in range(0, len(h_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [h_network[ith].mu_weight,
                                                  h_network[ith].rho_weight][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [h_network[0].weight]})

            for ith in range(0, len(c_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [c_network[ith].mu_weight,
                                                  c_network[ith].rho_weight][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [c_network[0].weight]})

            for ith in range(0, len(n_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [n_network[ith].mu_weight,
                                                  n_network[ith].rho_weight][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [n_network[0].weight]})

            for ith in range(0, len(o_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [o_network[ith].mu_weight,
                                                  o_network[ith].rho_weight][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [o_network[0].weight]})

        else:

            for ith in range(0, len(h_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [h_network[ith].mu_weight,
                                        h_network[ith].rho_weight][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [h_network[ith].mu_weight,
                                        h_network[ith].rho_weight][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [h_network[ith].mu_weight,
                                                      h_network[ith].rho_weight][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [h_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [h_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [h_network[ith].weight]})

            for ith in range(0, len(c_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [c_network[ith].mu_weight,
                                        c_network[ith].rho_weight][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [c_network[ith].mu_weight,
                                        c_network[ith].rho_weight][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [c_network[ith].mu_weight,
                                                      c_network[ith].rho_weight][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [c_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [c_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [c_network[ith].weight]})

            for ith in range(0, len(n_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [n_network[ith].mu_weight,
                                        n_network[ith].rho_weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [n_network[ith].mu_weight,
                                        n_network[ith].rho_weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [n_network[ith].mu_weight,
                                                      n_network[ith].rho_weight][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [n_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [n_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [n_network[ith].weight]})

            for ith in range(0, len(o_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [o_network[ith].mu_weight,
                                        o_network[ith].rho_weight][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [o_network[ith].mu_weight,
                                        o_network[ith].rho_weight][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [o_network[ith].mu_weight,
                                                      o_network[ith].rho_weight][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [o_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [o_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [o_network[ith].weight]})

        AdamW = torch.optim.AdamW(
            optimzable,
            lr=learning_rate,
        )

    elif bayesian is False:
        log.info("Network is not Bayesian. Attaching weights to AdamW optimizer")
        AdamW = torch.optim.AdamW(
            [
                # H networks
                {"params": [h_network[0].weight]},
                {"params": [h_network[2].weight], "weight_decay": 0.00001},
                {"params": [h_network[4].weight], "weight_decay": 0.000001},
                {"params": [h_network[6].weight]},
                # C networks
                {"params": [c_network[0].weight]},
                {"params": [c_network[2].weight], "weight_decay": 0.00001},
                {"params": [c_network[4].weight], "weight_decay": 0.000001},
                {"params": [c_network[6].weight]},
                # N networks
                {"params": [n_network[0].weight]},
                {"params": [n_network[2].weight], "weight_decay": 0.00001},
                {"params": [n_network[4].weight], "weight_decay": 0.000001},
                {"params": [n_network[6].weight]},
                # O networks
                {"params": [o_network[0].weight]},
                {"params": [o_network[2].weight], "weight_decay": 0.00001},
                {"params": [o_network[4].weight], "weight_decay": 0.000001},
                {"params": [o_network[6].weight]},
            ],
            lr=learning_rate,
        )

    else:
        log.error(
            "ERROR - conflicting arguments for bayesian and bayesian depth mean the "
            "optimizer cannot be defined")
        raise RuntimeError("Adam optimizer could not be created")

    return AdamW


def get_sgd_optimizer(h_network,
                      c_network,
                      n_network,
                      o_network,
                      bayesian_depth: int,
                      learning_rate: float,
                      use_ani_wd: bool=True,
                      bayesian: bool = False,
                      mu_opt: bool = True,
                      sig_opt: bool = True):
    """
    Function to get the appropiate SGD optimizer for the chosen ANI network
    :param h_network: torch.nn.Sequential - Network architecture for Hydrogen
    :param c_network: torch.nn.Sequential - Network architecture for Carbon
    :param n_network: torch.nn.Sequential - Network architecture for Nitrogen
    :param o_network: torch.nn.Sequential - Network architecture for Oxygen
    :param learning_rate: float - initial learning rate
    :param bayesian_depth: None or int - depth of bayesian network
    :param use_ani_wd: bool - use ani weight decay or not
    :param bayesian: bool - is a bayesian network
    :param mu_opt: bool - flag specifying whether to train weight means
    :param sig_opt: bool - flag specifying whether to train weight sigmas
    :returns: torch.optim.SGD - stochastic gradient descent optimizer
    """
    log = logging.getLogger(__name__)

    if mu_opt:
        start_ind = 0
    else:
        start_ind = 1
    if sig_opt:
        end_ind = 2
    else:
        end_ind = 1

    if bayesian_depth is None and bayesian is True:
        log.info("Network is Bayesian for all layers. Attaching bias for mu and rho to SGD optimizer")
        SGD = torch.optim.SGD(
            [
                # H networks
                {"params": [h_network[0].mu_bias, h_network[0].rho_bias][start_ind:end_ind]},
                {"params": [h_network[2].mu_bias, h_network[2].rho_bias][start_ind:end_ind]},
                {"params": [h_network[4].mu_bias, h_network[4].rho_bias][start_ind:end_ind]},
                {"params": [h_network[6].mu_bias, h_network[6].rho_bias][start_ind:end_ind]},
                # C networks
                {"params": [c_network[0].mu_bias, c_network[0].rho_bias][start_ind:end_ind]},
                {"params": [c_network[2].mu_bias, c_network[2].rho_bias][start_ind:end_ind]},
                {"params": [c_network[4].mu_bias, c_network[4].rho_bias][start_ind:end_ind]},
                {"params": [c_network[6].mu_bias, c_network[6].rho_bias][start_ind:end_ind]},
                # N networks
                {"params": [n_network[0].mu_bias, n_network[0].rho_bias][start_ind:end_ind]},
                {"params": [n_network[2].mu_bias, n_network[2].rho_bias][start_ind:end_ind]},
                {"params": [n_network[4].mu_bias, n_network[4].rho_bias][start_ind:end_ind]},
                {"params": [n_network[6].mu_bias, n_network[6].rho_bias][start_ind:end_ind]},
                # O networks
                {"params": [o_network[0].mu_bias, o_network[0].rho_bias][start_ind:end_ind]},
                {"params": [o_network[2].mu_bias, o_network[2].rho_bias][start_ind:end_ind]},
                {"params": [o_network[4].mu_bias, o_network[4].rho_bias][start_ind:end_ind]},
                {"params": [o_network[6].mu_bias, o_network[6].rho_bias][start_ind:end_ind]},
            ],
            lr=learning_rate,
        )

    elif bayesian_depth is not None and bayesian is True:
        log.info("Some layers are Bayesian. Attaching bias for mu and rho to SGD optimizer")
        optimzable = []

        # Assumption that you have [layer type, activation] * n hence only even
        # layers need updating
        if use_ani_wd is False:
            for ith in range(0, len(h_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [h_network[ith].mu_bias,
                                                  h_network[ith].rho_bias][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [h_network[0].weight]})

            for ith in range(0, len(c_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [c_network[ith].mu_bias,
                                                  c_network[ith].rho_bias][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [c_network[0].weight]})

            for ith in range(0, len(n_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [n_network[ith].mu_bias,
                                                  n_network[ith].rho_bias][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [n_network[0].weight]})

            for ith in range(0, len(o_network), 2):
                if ith >= bayesian_depth:
                    optimzable.append({"params": [o_network[ith].mu_bias,
                                                  o_network[ith].rho_bias][start_ind:end_ind]})
                else:
                    optimzable.append({"params": [o_network[0].weight]})

        else:

            for ith in range(0, len(h_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [h_network[ith].mu_bias,
                                        h_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [h_network[ith].mu_bias,
                                        h_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [h_network[ith].mu_bias,
                                                      h_network[ith].rho_bias][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [h_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [h_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [h_network[ith].weight]})

            for ith in range(0, len(c_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [c_network[ith].mu_bias,
                                        c_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [c_network[ith].mu_bias,
                                        c_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [c_network[ith].mu_bias,
                                                      c_network[ith].rho_bias][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [c_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [c_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [c_network[ith].weight]})

            for ith in range(0, len(n_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [n_network[ith].mu_bias,
                                        n_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [n_network[ith].mu_bias,
                                        n_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [n_network[ith].mu_bias,
                                                      n_network[ith].rho_bias][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [n_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [n_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [n_network[ith].weight]})

            for ith in range(0, len(o_network), 2):
                if ith >= bayesian_depth:
                    if ith == 2:
                        optimzable.append(
                            {"params": [o_network[ith].mu_bias,
                                        o_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [o_network[ith].mu_bias,
                                        o_network[ith].rho_bias][start_ind:end_ind],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [o_network[ith].mu_bias,
                                                      o_network[ith].rho_bias][start_ind:end_ind]})

                else:
                    if ith == 2:
                        optimzable.append(
                            {"params": [o_network[ith].weight],
                             "weight_decay": 0.00001})
                    elif ith == 4:
                        optimzable.append(
                            {"params": [o_network[ith].weight],
                             "weight_decay": 0.000001})
                    else:
                        optimzable.append({"params": [o_network[ith].weight]})

        SGD = torch.optim.SGD(
            optimzable,
            lr=learning_rate,
        )

    elif bayesian is False:
        log.info("Network is not Bayesian. Attaching bias to SGD optimizer")
        SGD = torch.optim.SGD(
            [
                # H networks
                {"params": [h_network[0].bias]},
                {"params": [h_network[2].bias]},
                {"params": [h_network[4].bias]},
                {"params": [h_network[6].bias]},
                # C networks
                {"params": [c_network[0].bias]},
                {"params": [c_network[2].bias]},
                {"params": [c_network[4].bias]},
                {"params": [c_network[6].bias]},
                # N networks
                {"params": [n_network[0].bias]},
                {"params": [n_network[2].bias]},
                {"params": [n_network[4].bias]},
                {"params": [n_network[6].bias]},
                # O networks
                {"params": [o_network[0].bias]},
                {"params": [o_network[2].bias]},
                {"params": [o_network[4].bias]},
                {"params": [o_network[6].bias]},
            ],
            lr=learning_rate,
        )

    else:
        log.error(
            "ERROR - conflicting arguments for bayesian and bayesian depth mean the "
            "optimizer cannot be defined")
        raise RuntimeError("SGD optimizer could not be created")

    return SGD


def get_default_ani_optimizers(ani_class: network_classes.ANI,
                               learning_rate: float,
                               bayesian_depth: int,
                               mu_opt: bool = True,
                               sig_opt: bool = True):
    """
    Function to get optimizers for ANI model
    :param ani_class: network_classes.ANI - default ani np
    :param learning_rate: float - initial learning rate
    :param bayesian_depth: int - Depth of Bayesian network
    :returns: (torch.optim.AdamW, torch.optim.SGD) ANI network Optimizers
    """

    log = logging.getLogger(__name__)

    if ani_class.default_ani is True:
        log.info("Using default ANI optimizers")

        AdamW = get_adamw_optimizer(ani_class.h_network,
                                    ani_class.c_network,
                                    ani_class.n_network,
                                    ani_class.o_network,
                                    learning_rate=learning_rate,
                                    bayesian_depth=bayesian_depth,
                                    bayesian=ani_class.isbayesian,
                                    mu_opt=mu_opt,
                                    sig_opt=sig_opt
                                    )


        SGD = get_sgd_optimizer(ani_class.h_network,
                                ani_class.c_network,
                                ani_class.n_network,
                                ani_class.o_network,
                                learning_rate=learning_rate,
                                bayesian_depth=bayesian_depth,
                                bayesian=ani_class.isbayesian,
                                mu_opt=mu_opt,
                                sig_opt=sig_opt
                                )

        return AdamW, SGD

    else:
        log.error(f"ERROR - function {inspect.stack()[0][3]} is only for use with default ANI networks of elements"
                  f" hydrogen, carbon, nitrogen and oxygen. The current network has not loaded the default ANI networks"
                  f" so needs to have the optimizers specifically set. You can try get_generic_ani_optimizers().")

def get_generic_ani_optimizers(ani_class: network_classes.ANI,
                               learning_rate: float,
                               momentum: float = 0.0,
                               adam_op: bool = True,
                               sgd_op: bool = False):
    """
    Function to get generic optimizers for a non-standard ANI model
    :param ani_class: network_classes.ANI - default ani np
    :param learning_rate: float - initial learning rate
    :param momentum: float - in the SGD optimimzer use momentum and initialize the value of the momentum
    :param adam_op: bool - use the AdamW optimizer
    :param sgd_op: bool - use the SGD optimizer
    :return:
    """

    log = logging.getLogger(__name__)

    log.info(f"Getting generic optimizer instances for non-standard ANI. You may be better doing this yourself! "
             f"Check carefully.")

    if adam_op is True:
        optimizer = torch.optim.AdamW(ani_class.model.parameters(), lr=learning_rate)

    elif sgd_op is True:
        optimizer = torch.optim.SGD(ani_class.model.parameters(), lr=learning_rate, momentum=momentum)

    else:
        log.error(f"ERROR - function {inspect.stack()[0][3]} provide a general AdamW or SGD option both are false "
                  f"please pick one.")
        return None

    return optimizer

def get_default_schedulers(adamw: torch.optim.AdamW,
                           sgd: torch.optim.SGD,
                           adamw_factor: float = 0.1,
                           adamw_patience: int = 100,
                           adamw_threshold: int = 0,
                           sgd_factor: float = 0.1,
                           sgd_patience: int = 100,
                           sgd_threshold: int = 0):
    """
    Function to set schedulers for training. Note you must pass in the optimizers for this to run correctly.
    :param sgd: torch.optim.SGD - optimizer instance for SGD
    :param adamw: torch.optim.AdamW - instance of the AdamW optimizer
    :param adamw_factor: float - set the reduction factor for AdamW scheduler
    :param adamw_patience: int - set the delay in improvement for this many steps in the scheduler
    :param adamw_threshold: int - threshold value for AdamW scheduler
    :param sgd_factor: float - set the reduction factor for sgd scheduler
    :param sgd_patience: int - set the delay in improvement for this many steps in the scheduler
    :param sgd_threshold: int - threshold value for sgd scheduler
    :return: (torch.optim.AdamW, torch.optim.SGD) - returns the initialized schedulers
    """

    sgd_ani_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        sgd, factor=sgd_factor, patience=sgd_patience, threshold=sgd_threshold
    )

    adamw_ani_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        adamw, factor=adamw_factor, patience=adamw_patience, threshold=adamw_threshold
    )

    return adamw_ani_scheduler, sgd_ani_scheduler

@torch.no_grad()
def validate_bnn(model: torchani.nn.Sequential,
                 val_data: torchani.data.TransformableIterable,
                 epoch: int,
                 mc_runs: int = 2,
                 kl_weight: float = 1.0,
                 ):
    """
    Function to run validation with Monte Carlo with gradient calculation disabled
    :param model: torch.nn.Sequential - ANI BNN network model
    :param val_data: torchani.data.TransformableIterable - validation data
    :param epoch: int - epoch number
    :param mc_runs: int - number of mote carlo samples
    :param kl_weight: float - weighting for the kl divergence
    :returns: float - Root Mean Square Error kcal/mol
    """

    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_sum = torch.nn.MSELoss(reduction="sum")
    total_mse = 0.0
    count = 0
    scaled_kl_ = []

    predicted_energies_plot = []
    true_energies_plot = []
    predicted_energies_sigma_plot = []

    model.train(False)
    log.info("\nValidating no gradients ......")
    for properties in val_data:
        species = properties["species"].to(device)
        coordinates = properties["coordinates"].to(device).float()
        true_energies = properties["energies"].to(device).float()
        output_ = []
        kl_ = []
        batch_size = len(species)
        log.debug("Validation no gradients batch size: {}".format(batch_size))
        for mc_run in range(mc_runs):
            _, predicted_energies = model((species, coordinates))
            kl = get_kl_loss(model)
            output_.append(predicted_energies)
            kl_.append(kl)
        kl = torch.mean(torch.stack(kl_), dim=0)

        predicted_energies = torch.mean(torch.stack(output_), dim=0)
        predicted_energies_sigma = torch.stack(output_).std(dim=0)

        true_energies_plot = true_energies_plot + true_energies.clone().detach().tolist()
        predicted_energies_plot = predicted_energies_plot + predicted_energies.clone().detach().tolist()
        predicted_energies_sigma_plot = predicted_energies_sigma_plot + predicted_energies_sigma.clone().detach().tolist()

        scaled_kl_.append(kl / batch_size)
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    scaled_kl = torch.mean(torch.stack(scaled_kl_), dim=0)
    log.info("scaled kl: {}".format(scaled_kl.item()))
    log.info("weighted by {} and scaled kl: {}".format(kl_weight, scaled_kl.item() * kl_weight))
    rmse_ha = math.sqrt(total_mse / count)
    log.info("RMSE: {} kcal/mol".format(hartree2kcalmol(math.sqrt(total_mse / count))))
    model.train(True)
    log.info("Validated\n")

    if epoch % 1000 == 0:
        log.info("Plotting errors and energies")
        errors = np.array([p - t for p, t in zip(predicted_energies_plot, true_energies_plot)])
        abs_errors = np.absolute(errors)

        df = pd.DataFrame(data=np.array([errors, abs_errors, predicted_energies_sigma_plot, true_energies_plot,
                                         predicted_energies_plot]).T,
                          columns=["errors", "absolute_errors", "pred_energy_sigma", "true_energies",
                                   "predicted_energies" ])

        df.to_csv("errors_and_uncertaininty_epoch_{}.csv".format(epoch), index=False)

        ax_plt = df.plot(x="pred_energy_sigma", y="errors", kind="scatter", figsize=(15, 15), grid=True,
                         xlabel="Predicted Energy Sigma (Ha)",
                         ylabel="Predicted Energy Error (predicted - true) (Ha)")
        ax_plt.figure.savefig("uncertainity_against_error_epoch_{}.png".format(epoch))

        ax_plt_abs = df.plot(x="pred_energy_sigma", y="absolute_errors", kind="scatter", figsize=(15, 15), grid=True,
                             xlabel="Predicted Energy Sigma (Ha)",
                             ylabel="Predicted Absolute eEnergy Error (predicted - true) (Ha)")
        ax_plt_abs.figure.savefig("uncertainity_against_abs_error_epoch_{}.png".format(epoch))

        fig = plt.figure(figsize=(15, 15))
        plt.plot(true_energies_plot,
                 predicted_energies_plot,
                 "bo",
                 label="True Vs predicted energies (Ha) RMSE: {} Ha {} kcal/mol".format(rmse_ha,
                                                                                        hartree2kcalmol(rmse_ha))
                 )
        plt.legend()
        plt.xlabel("True Energies (Ha)")
        plt.ylabel("Predicted Energies (Ha)")
        plt.grid(True)
        plt.savefig("energies_against_predictions.png")

        fig = plt.figure(figsize=(15, 15))
        plt.errorbar(true_energies_plot,
                     predicted_energies_plot,
                     yerr=predicted_energies_sigma_plot,
                     fmt="bo",
                     ecolor="g",
                     elinewidth=1.2,
                     capsize=0.1,
                     capthick=1.2,
                     label="True Vs predicted energies (Ha) with uncert RMSE: {} Ha {} kcal/mol".format(rmse_ha,
                                                                                                        hartree2kcalmol(
                                                                                                            rmse_ha))
                     )
        plt.legend()
        plt.xlabel("True Energies (Ha)")
        plt.ylabel("Predicted Energies and uncert (Ha)")
        plt.grid(True)
        plt.savefig("energies_against_predictions_with_y_uncertainty.png")


        log_likelihoods = [log_likelihood(p, p_sigma, t) for p, p_sigma, t in zip(predicted_energies_plot,
                                                                                  predicted_energies_sigma_plot,
                                                                                  true_energies_plot)]
        x = np.array([ith for ith in range(len(predicted_energies_plot))])
        ll_df = pd.DataFrame([x, log_likelihoods]).transpose()
        ll_df.columns = ["index", "ln likelihood"]
        ll_df.to_csv(f"log_likelihoods_epoch_{epoch}.csv")

        uncertainty_mets = get_uncertainity_metrics(np.array(predicted_energies_plot),
                                                    np.array(predicted_energies_sigma_plot),
                                                    np.array(true_energies_plot),
                                                    x=x,
                                                    epoch=epoch,
                                                    plot=False)

        log.info(f"Uncertainty metrics for epoch {epoch}\n{uncertainty_mets}")

    return hartree2kcalmol(math.sqrt(total_mse / count)) + scaled_kl.item() * kl_weight

@torch.no_grad()
def validate_dnn(model: torchani.nn.Sequential,
                 val_data: torchani.data.TransformableIterable,
                 epoch: Union[int, str, None] = None,
                 plot: bool = True
                 ):
    """
    Function to run validation with Monte Carlo with gradient calculation
    :param model: torch.nn.Sequential - ANI BNN network model
    :param val_data: torchani.data.TransformableIterable - validation data
    :returns: float - Root Mean Square Error kcal/mol
    """

    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_sum = torch.nn.MSELoss(reduction="sum")
    total_mse = 0.0
    count = 0
    predicted_energies_plot = []
    true_energies_plot = []
    model.train(False)
    log.info("\nValidating ......")

    for properties in val_data:
        species = properties["species"].to(device)
        coordinates = properties["coordinates"].to(device).float()
        true_energies = properties["energies"].to(device).float()
        _, predicted_energies = model((species, coordinates))
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]

    rmse_ha = math.sqrt(total_mse / count)

    if plot == True:
        log.info("Plotting errors and energies")
        errors = np.array([p - t for p, t in zip(predicted_energies_plot, true_energies_plot)])
        abs_errors = np.absolute(errors)

        df = pd.DataFrame(data=np.array([errors, abs_errors, true_energies_plot,
                                         predicted_energies_plot]).T,
                          columns=["errors", "absolute_errors", "true_energies",
                                   "predicted_energies" ])

        df.to_csv("errors_epoch_{}.csv".format(epoch), index=False)

        fig = plt.figure(figsize=(15, 15))
        plt.plot(true_energies_plot,
                 predicted_energies_plot,
                 "bo",
                 label="True Vs predicted energies (Ha) RMSE: Ha {} kcal/mol".format(rmse_ha,
                                                                                      hartree2kcalmol(rmse_ha)
                                                                                      )
                 )
        plt.legend()
        plt.xlabel("True Energies (Ha)")
        plt.ylabel("Predicted Energies (Ha)")
        plt.grid(True)
        plt.savefig("energies_against_predictions.png")

    log.info("RMSE: {} kcal/mol".format(hartree2kcalmol(math.sqrt(total_mse / count))))
    model.train(True)
    log.info("Validated\n")

    return hartree2kcalmol(math.sqrt(total_mse / count))

def train(ani_class: network_classes.ANI,
          train_data: torchani.data.TransformableIterable,
          val_data: torchani.data.TransformableIterable,
          test_data: torchani.data.TransformableIterable,
          checkpoint_filename: str = "ani_model.pt",
          learning_curve_csv_filename: str = "learning_curve.csv",
          max_epochs: int = 2,
          early_stopping_learning_rate: float = 1.0e-9,
          train_mc_runs: int = 2,
          test_mc_runs: int = 2,
          kl_weight: float = 1.0,
          learning_rate: float = 0.00001,
          bayesian_depth: int = None,
          mu_opt: bool = True,
          sig_opt: bool = True,
          lml: bool = False,
          use_schedulers: bool = False,
          checkpoint_dnn: str = None,
          checkpoint_bnn: str = None,
          reset_lr: bool = False,
          min_rmse: float = 0.0
          ):
    """
    Function to train ANI model. This function spilt off to train DNN or BNN with the appropriate use of monte carlo
    sampling etc for a BNN.

    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param train_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param test_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param kl_weight: float - weight for the KL term
    :param learning_rate: float - learning rate to start
    :param bayesian_depth: int or None - the depth of Bayesian layers, note start at the end and works backwards
    :param mu_opt: bool - flag specifying whether to train weight means
    :param sig_opt: bool - flag specifying whether to train weight sigmas
    :param lml: bool = whether bnn should use lml loss in place of rmse
    :param use_schedulers: bool - Use pytorch parameter schedulers
    :param checkpoint_dnn: str - file of stored previous versions of schedulers and epoch to load for a DNN
    :param checkpoint_bnn: str - file of stored previous versions of schedulers and epoch to load for a BNN
    :return: float - loss, note model is stored in the ANI class
    """

    log = logging.getLogger(__name__)

    if ani_class.isbayesian is True:
        log.info("Training BNN.....\n")

        loss = train_bnn(ani_class,
                  train_data,
                  val_data,
                  checkpoint_filename=checkpoint_filename,
                  learning_curve_csv_filename=learning_curve_csv_filename,
                  max_epochs=max_epochs,
                  early_stopping_learning_rate=early_stopping_learning_rate,
                  train_mc_runs=train_mc_runs,
                  test_mc_runs=test_mc_runs,
                  kl_weight=kl_weight,
                  learning_rate=learning_rate,
                  bayesian_depth=bayesian_depth,
                  use_schedulers=use_schedulers,
                  checkpoint=checkpoint_bnn,
                  mu_opt=mu_opt,
                  sig_opt=sig_opt,
                  lml=lml)

        return loss

    else:
        log.info("Training DNN .....\n")

        loss = train_dnn(ani_class,
                  train_data,
                  val_data,
                  test_data,
                  checkpoint_filename=checkpoint_filename,
                  learning_curve_csv_filename=learning_curve_csv_filename,
                  max_epochs=max_epochs,
                  early_stopping_learning_rate=early_stopping_learning_rate,
                  learning_rate=learning_rate,
                  bayesian_depth=bayesian_depth,
                  use_schedulers=use_schedulers,
                  reset_lr = reset_lr,
                  checkpoint=checkpoint_dnn,
                  min_rmse = min_rmse)

        return loss

def train_dnn(ani_class: network_classes.ANI,
              train_data: torchani.data.TransformableIterable,
              val_data: torchani.data.TransformableIterable,
              test_data: torchani.data.TransformableIterable,
              checkpoint_filename = "ani_model.pt",
              learning_curve_csv_filename = "learning_curve.csv",
              max_epochs: int = 2,
              early_stopping_learning_rate : float = 1.0e-9,
              learning_rate: float = 0.00001,
              bayesian_depth: Union[int, None] = None,
              use_schedulers: bool = False,
              reset_lr: bool = False,
              checkpoint: str = None,
              min_rmse: float = 0.0 
              ):
    """
    Function to train DNN ANI model
    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param learning_rate: float - learning rate to start
    :param bayesian_depth: int or None - the depth of Bayesian layers, note start at the end and works backwards
    :param use_schedulers: bool - Use pytorch parameter schedulers
    :param checkpoint: str - file of stored previous versions of schedulers and epoch to load for a DNN
    :return: float - loss, note the model is stored in the ANI class
    """
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    AdamW, SGD = get_default_ani_optimizers(ani_class,
                                            learning_rate=learning_rate,
                                            bayesian_depth=bayesian_depth)

    if checkpoint is not None:
        checkp = torch.load(checkpoint, map_location=device)
        if reset_lr is False:
          if "AdamW" in checkp.keys():
                AdamW.load_state_dict(checkp["AdamW"])
          if "SGD" in checkp.keys():
                SGD.load_state_dict(checkp["SGD"])
        if "epoch" in checkp.keys():
            epoch = checkp["epoch"]
        else:
            epoch = 0
    else:
        epoch = 0

    if ani_class.sgd_ani_scheduler is not None and ani_class.adamw_ani_scheduler is not None and use_schedulers is True:
        log.info("Note ANI class has schedulers specified using these schedulers")
        SGD_scheduler = ani_class.sgd_ani_scheduler
        AdamW_scheduler = ani_class.adamw_ani_scheduler
        loss = train_dnn_with_schedulers(ani_class,
                                  train_data,
                                  val_data,
                                  test_data,
                                  AdamW,
                                  SGD,
                                  AdamW_scheduler,
                                  SGD_scheduler,
                                  checkpoint_filename=checkpoint_filename,
                                  learning_curve_csv_filename=learning_curve_csv_filename,
                                  max_epochs=max_epochs,
                                  early_stopping_learning_rate=early_stopping_learning_rate,
                                  epoch=epoch,
                                  min_rmse= min_rmse)

    elif use_schedulers is True:
        log.info("Note 'use_schedulers' is True using default ANI schedulers")
        AdamW_scheduler, SGD_scheduler = get_default_schedulers(AdamW, SGD)
        loss = train_dnn_with_schedulers(ani_class,
                                  train_data,
                                  val_data,
                                  test_data,
                                  AdamW,
                                  SGD,
                                  AdamW_scheduler,
                                  SGD_scheduler,
                                  checkpoint_filename=checkpoint_filename,
                                  learning_curve_csv_filename=learning_curve_csv_filename,
                                  max_epochs=max_epochs,
                                  early_stopping_learning_rate=early_stopping_learning_rate,
                                  epoch=epoch,
                                  min_rmse= min_rmse)

    else:
        log.info("Note 'use_schedulers' is False will not use Pytorch parameter schedulers")
        loss = train_dnn_without_schedulers(ani_class,
                                     train_data,
                                     val_data,
                                     AdamW,
                                     SGD,
                                     checkpoint_filename=checkpoint_filename,
                                     learning_curve_csv_filename=learning_curve_csv_filename,
                                     max_epochs=max_epochs,
                                     early_stopping_learning_rate=early_stopping_learning_rate,
                                     epoch=epoch)

    return loss

def train_dnn_without_schedulers(ani_class: network_classes.ANI,
                                 train_data: torchani.data.TransformableIterable,
                                 val_data: torchani.data.TransformableIterable,
                                 AdamW: torch.optim.AdamW,
                                 SGD: torch.optim.SGD,
                                 checkpoint_filename:str = "ani_model.pt",
                                 learning_curve_csv_filename:str = "learning_curve.csv",
                                 max_epochs: int = 2,
                                 early_stopping_learning_rate : float = 1.0e-9,
                                 epoch: int = 0
                                 ):
    """
    Function to train DNN ANI without the use of learning rate schedulers

    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param AdamW: torch.optim.AdamW - instance for the AdamW optimizer
    :param SGD: torch.optim.SGD - instance for the SGD optimizer
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param epoch: int - epoch number to start from
    :return: float - loss, note model is stored in the ANI class
    """
    log = logging.getLogger(__name__)

    latest_checkpoint = f"latest_{checkpoint_filename}"
    best_model_checkpoint = f"best_{checkpoint_filename}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ani_class.model.to(device)

    # Replace scheduler with this dictionary so you can keep track of the variables still
    AdamW_monitor = {}
    AdamW_monitor["last_epoch"] = epoch
    AdamW_monitor["best"] = 10000

    log.info('train data length:{}'.format(len(train_data)))

    tensorboard = torch.utils.tensorboard.SummaryWriter()

    if os.path.isfile(learning_curve_csv_filename):
        learning_curve_df = pd.read_csv(learning_curve_csv_filename)
    else:
        learning_curve_df = pd.DataFrame(columns=["Epoch", "lr", "RMSE"])

    mse = torch.nn.MSELoss(reduction="none")

    log.info("Training starting from epoch:{}".format(AdamW_monitor["last_epoch"]))
    log.info("Early stopping rate is {}".format(early_stopping_learning_rate))
    for epoch in range(AdamW_monitor["last_epoch"], max_epochs):
        AdamW_monitor["last_epoch"] = epoch
        rmse = validate_dnn(ani_class.model, val_data, epoch)
        log.info("Validation Loss (RMSE): {} at Epoch:{}".format(rmse, AdamW_monitor["last_epoch"]))

        learning_rate = AdamW.param_groups[0]["lr"]

        learning_curve_df = learning_curve_df.append(
            {"Epoch": AdamW_monitor["last_epoch"],
             "lr": AdamW.param_groups[0]["lr"],
             "RMSE": rmse
             },
            ignore_index=True,
        )

        learning_curve_df.to_csv(learning_curve_csv_filename, index=False)

        if learning_rate < early_stopping_learning_rate:
            break

        # best checkpoint
        if rmse < AdamW_monitor["best"]:
            if ani_class.isparallel is False:
                torch.save(ani_class.model[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            else:
                torch.save(ani_class.model.module[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            AdamW_monitor["best"] = rmse

        tensorboard.add_scalar("validation_rmse",
                               rmse,
                               AdamW_monitor["last_epoch"]
                               )

        tensorboard.add_scalar(
            "best_validation_rmse",
            AdamW_monitor["best"],
            AdamW_monitor["last_epoch"]
        )

        tensorboard.add_scalar(
            "learning_rate",
            learning_rate,
            AdamW_monitor["last_epoch"]
        )

        for i, properties in tqdm.tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc="epoch {}".format(AdamW_monitor["last_epoch"]),
        ):

            species = properties["species"].to(device)
            # for auto diffrentation
            if ani_class.forces is True:
                coordinates = properties["coordinates"].to(device).float().requires_grad_(True)
            else:
                coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            batch_size = len(species)
            log.debug("Batch size: {}".format(batch_size))

            _, predicted_energies = ani_class.model((species, coordinates))

            if ani_class.forces is True:
                true_forces = properties['forces'].to(device).float()

                # We can use torch.autograd.grad to compute force. Remember to
                # create graph so that the loss of the force can contribute to
                # the gradient of parameters, and also to retain graph so that
                # we can backward through it a second time when computing gradient
                # w.r.t. parameters.
                # https://aiqm.github.io/torchani/examples/nnp_training_force.html
                forces = - \
                torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

                # Now the total loss has two parts, energy loss and force loss
                energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
                loss = energy_loss + ani_class.force_scalar * force_loss
                log.debug(f"energy loss {energy_loss} force loss {force_loss} force scalar {ani_class.force_scalar} "
                         f"total loss {loss}")

            else:
                loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                log.debug(f"energy loss {loss} total loss {loss}")

            log.debug("Training Loss: {}".format(loss))
            AdamW.zero_grad()
            SGD.zero_grad()
            loss.backward()
            AdamW.step()
            SGD.step()

            # write current batch loss to TensorBoard
            tensorboard.add_scalar(
                "batch_loss", loss, AdamW_monitor["last_epoch"] * len(train_data) + i
            )
        log.info('Saving model to: {}'.format(latest_checkpoint))
        if ani_class.isparallel is False:
            torch.save(
                {
                    "nn": ani_class.model[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )
        else:
            torch.save(
                {
                    "nn": ani_class.model.module[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )

        if epoch == max_epochs - 1:
            rmse = validate_dnn(ani_class.model, val_data)
            log.info("Validation Loss (RMSE): {} at Epoch:{}".format(rmse, AdamW_scheduler.last_epoch))

    return AdamW_monitor["best"]

def train_dnn_with_schedulers(ani_class: network_classes.ANI,
                              train_data: torchani.data.TransformableIterable,
                              val_data: torchani.data.TransformableIterable,
                              test_data: torchani.data.TransformableIterable,
                              AdamW,
                              SGD,
                              AdamW_scheduler,
                              SGD_scheduler,
                              checkpoint_filename: str = "ani_model.pt",
                              learning_curve_csv_filename: str = "learning_curve.csv",
                              max_epochs: int = 2,
                              early_stopping_learning_rate : float = 1.0e-9,
                              epoch: int = 0,
                              min_rmse: float = 0
                              ):
    """
    Function to train DNN ANI with the use of learning rate schedulers

    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param AdamW: torch.optim.AdamW - instance for the AdamW optimizer
    :param SGD: torch.optim.SGD - instance for the SGD optimizer
    :param AdamW_scheduler: torch.optim.lr_scheduler - instance for learn rate scheduler for the AdamW optimizer
    :param SGD_scheduler: torch.optim.lr_scheduler - instance for learn rate scheduler for the SGD optimizer
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param epoch: int - epoch number to start from
    :return: float - loss, note model is stored in the ANI class
    """
    log = logging.getLogger(__name__)

    latest_checkpoint = f"latest_{checkpoint_filename}"
    best_model_checkpoint = f"best_{checkpoint_filename}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ani_class.model.to(device)

    log.info('train data length:{}'.format(len(train_data)))

    tensorboard = torch.utils.tensorboard.SummaryWriter()

    if os.path.isfile(learning_curve_csv_filename):
        learning_curve_df = pd.read_csv(learning_curve_csv_filename)
    else:
        learning_curve_df = pd.DataFrame(columns=["Epoch", "lr", "RMSE"])

    mse = torch.nn.MSELoss(reduction="none")

    AdamW_scheduler.last_epoch = epoch

    log.info("Training starting from epoch:{}".format(AdamW_scheduler.last_epoch))
    log.info("Early stopping rate is {}".format(early_stopping_learning_rate))
    for epoch in range(AdamW_scheduler.last_epoch, max_epochs):
        AdamW_scheduler.last_epoch = epoch
        rmse = validate_dnn(ani_class.model, val_data)
        log.info("Validation Loss (RMSE): {} at Epoch:{}".format(rmse, AdamW_scheduler.last_epoch))

        learning_rate = AdamW.param_groups[0]["lr"]

        learning_curve_df = learning_curve_df.append(
            {"Epoch": AdamW_scheduler.last_epoch,
             "lr": AdamW.param_groups[0]["lr"],
             "RMSE": rmse
             },
            ignore_index=True,
        )

        learning_curve_df.to_csv(learning_curve_csv_filename, index=False)

        if learning_rate < early_stopping_learning_rate:
            break

        if rmse < min_rmse:
            break

        # best checkpoint
        if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
            if ani_class.isparallel is False:
                torch.save(ani_class.model[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            else:
                torch.save(ani_class.model.module[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            AdamW_scheduler.best = rmse

        AdamW_scheduler.step(rmse)
        SGD_scheduler.step(rmse)

        tensorboard.add_scalar("validation_rmse",
                               rmse,
                               AdamW_scheduler.last_epoch
                               )

        tensorboard.add_scalar(
            "best_validation_rmse",
            AdamW_scheduler.best,
            AdamW_scheduler.last_epoch
        )

        tensorboard.add_scalar(
            "learning_rate",
            learning_rate,
            AdamW_scheduler.last_epoch
        )

        for i, properties in tqdm.tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc="epoch {}".format(AdamW_scheduler.last_epoch),
        ):

            species = properties["species"].to(device)
            # for auto diffrentation
            if ani_class.forces is True:
                coordinates = properties["coordinates"].to(device).float().requires_grad_(True)
            else:
                coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            batch_size = len(species)
            log.debug("Batch size: {}".format(batch_size))

            _, predicted_energies = ani_class.model((species, coordinates))

            if ani_class.forces is True:
                true_forces = properties['forces'].to(device).float()

                # We can use torch.autograd.grad to compute force. Remember to
                # create graph so that the loss of the force can contribute to
                # the gradient of parameters, and also to retain graph so that
                # we can backward through it a second time when computing gradient
                # w.r.t. parameters.
                # https://aiqm.github.io/torchani/examples/nnp_training_force.html
                forces = - \
                torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

                # Now the total loss has two parts, energy loss and force loss
                energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
                loss = energy_loss + ani_class.force_scalar * force_loss
                log.debug(f"energy loss {energy_loss} force loss {force_loss} force scaler {ani_class.force_scalar} "
                         f"total loss {loss}")

            else:
                loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                log.debug(f"energy loss {loss} total loss {loss}")

            log.debug("Training Loss: {}".format(loss))
            AdamW.zero_grad()
            SGD.zero_grad()
            loss.backward()
            AdamW.step()
            SGD.step()

            # write current batch loss to TensorBoard
            tensorboard.add_scalar(
                "batch_loss", loss, AdamW_scheduler.last_epoch * len(train_data) + i
            )
        log.info('Saving model to: {}'.format(latest_checkpoint))
        if ani_class.isparallel is False:
            torch.save(
                {
                    "nn": ani_class.model[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )
        else:
            torch.save(
                {
                    "nn": ani_class.model.module[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )

        if epoch == max_epochs - 1:
            rmse = validate_dnn(ani_class.model, val_data)
            log.info("Validation Loss (RMSE): {} at Epoch:{}".format(rmse, AdamW_scheduler.last_epoch))


        # if epoch % 1000 == 0:
        #     ani_class.model.to(device)
        #     network_test.test(ani_class,
        #                 test_data,
        #               mc_runs=0,
        #               plot=True,
        #               plot_name_unique_prepend="epoch_"+str(epoch)+"_"
        #               )

    return AdamW_scheduler.best

def train_bnn(ani_class: network_classes.ANI,
              train_data: torchani.data.TransformableIterable,
              val_data: torchani.data.TransformableIterable,
              checkpoint_filename = "ani_model.pt",
              learning_curve_csv_filename = "learning_curve.csv",
              max_epochs: int = 2,
              early_stopping_learning_rate : float = 1.0e-9,
              train_mc_runs: int = 2,
              test_mc_runs: int = 2,
              kl_weight: float = 1.0,
              learning_rate: float = 0.00001,
              bayesian_depth: int = None,
              use_schedulers: bool = False,
              checkpoint: str = None,
              mu_opt: bool = True,
              sig_opt: bool = True,
              lml: bool = False
              ):
    """
    Function to train BNN ANI model

    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param train_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param test_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param kl_weight: float - weight for the KL term
    :param learning_rate: float - learning rate to start
    :param bayesian_depth: int or None - the depth of Bayesian layers, note start at the end and works backwards
    :param use_schedulers: bool - Use pytorch parameter schedulers
    :param checkpoint: str - file of stored previous versions of schedulers and epoch to load for a BNN
    :param mu_opt: bool - flag specifying whether to train weight means
    :param sig_opt: bool - flag specifying whether to train weight sigmas
    :param lml: bool = whether to use lml loss in place of rmse
    :return: float - loss, note model is stored in the ANI class
    """
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AdamW, SGD = get_default_ani_optimizers(ani_class,
                                            learning_rate=learning_rate,
                                            bayesian_depth=bayesian_depth,
                                            mu_opt=mu_opt,
                                            sig_opt=sig_opt)

    log.debug(AdamW.param_groups)
    log.debug(SGD.param_groups)

    if checkpoint is not None:
        checkp = torch.load(checkpoint, map_location=device)
        if "AdamW" in checkp.keys() and mu_opt and sig_opt:
            AdamW.load_state_dict(checkp["AdamW"])
            AdamW.param_groups[0]["lr"] = learning_rate
        if "SGD" in checkp.keys() and mu_opt and sig_opt:
            SGD.load_state_dict(checkp["SGD"])
            SGD.param_groups[0]["lr"] = learning_rate
        if "epoch" in checkp.keys():
            epoch = checkp["epoch"]
        else:
            epoch = 0
    else:
        epoch = 0

    if ani_class.sgd_ani_scheduler is not None and ani_class.adamw_ani_scheduler is not None and use_schedulers is True:
        log.info("Note ANI class has schedulers specified using these schedulers")
        SGD_scheduler = ani_class.sgd_ani_scheduler
        AdamW_scheduler = ani_class.adamw_ani_scheduler
        loss = train_bnn_with_schedulers(ani_class,
                                  train_data,
                                  val_data,
                                  AdamW,
                                  SGD,
                                  AdamW_scheduler,
                                  SGD_scheduler,
                                  checkpoint_filename=checkpoint_filename,
                                  learning_curve_csv_filename=learning_curve_csv_filename,
                                  max_epochs=max_epochs,
                                  early_stopping_learning_rate=early_stopping_learning_rate,
                                  train_mc_runs=train_mc_runs,
                                  test_mc_runs=test_mc_runs,
                                  kl_weight=kl_weight,
                                  epoch=epoch,
                                  lml=lml)

    elif use_schedulers is True:
        log.info("Note 'use_schedulers' is True using default ANI schedulers")
        AdamW_scheduler, SGD_scheduler = get_default_schedulers(AdamW, SGD)
        loss = train_bnn_with_schedulers(ani_class,
                                  train_data,
                                  val_data,
                                  AdamW,
                                  SGD,
                                  AdamW_scheduler,
                                  SGD_scheduler,
                                  checkpoint_filename=checkpoint_filename,
                                  learning_curve_csv_filename=learning_curve_csv_filename,
                                  max_epochs=max_epochs,
                                  early_stopping_learning_rate=early_stopping_learning_rate,
                                  train_mc_runs=train_mc_runs,
                                  test_mc_runs=test_mc_runs,
                                  kl_weight=kl_weight,
                                  epoch=epoch,
                                  lml=lml)

    else:
        log.info("Note 'use_schedulers' is False will not use Pytorch parameter schedulers")
        loss = train_bnn_without_schedulers(ani_class,
                                     train_data,
                                     val_data,
                                     AdamW,
                                     SGD,
                                     checkpoint_filename=checkpoint_filename,
                                     learning_curve_csv_filename=learning_curve_csv_filename,
                                     max_epochs=max_epochs,
                                     early_stopping_learning_rate=early_stopping_learning_rate,
                                     train_mc_runs=train_mc_runs,
                                     test_mc_runs=test_mc_runs,
                                     kl_weight=kl_weight,
                                     epoch=epoch,
                                     lml=lml)

    return loss


def mse_e_loss(batch_pred_e, batch_ref_e, num_atoms, loss_f):
    u = torch.mean(batch_pred_e, axis=0)
    batch_mse = loss_f(u, batch_ref_e)
    return torch.mean(batch_mse/ num_atoms.sqrt())


def mse_f_loss(batch_pred_f, batch_ref_f, num_atoms, loss_f):
    batch_mse = loss_f(batch_pred_f, batch_ref_f).sum(dim=(1, 2))
    return torch.mean(batch_mse/ num_atoms)


#def lml_loss_cov(batch_pred_e, batch_ref_e, sigma=1e-7, log=None):
#    #print(batch_pred_e.shape, batch_ref_e.shape)
#    device = batch_pred_e.device
#    batch_size = batch_ref_e.shape[0]
#    u = batch_pred_e.mean(axis=0)
#    C = batch_pred_e.t().cov() + torch.diag(sigma*torch.ones(u.shape[0], device=device))
#
#    #print(u.shape, C.shape)
#    delta_e = (batch_ref_e - u)[None,:]
#    d = torch.log(C.det())
#    p = delta_e @ torch.inverse(C) @ delta_e.t()
#    c = batch_size * np.log(2*np.pi)
#
#    if log is not None:
#        log.debug(f'd: {d}')
#        log.debug(f'p: {p}')
#        log.debug(f'c: {c}')
#
#    lml = -0.5*(d + p + c)
#    return -lml

def lml_loss(batch_pred_e, batch_ref_e, num_atoms, log=None):
    device = batch_pred_e.device
    batch_size = batch_ref_e.shape[0]

    u = batch_pred_e.mean(axis=0)
    v = batch_pred_e.var(axis=0)

    delta_e = (batch_ref_e - u)
    p = -0.5*(delta_e**2) / v
    c = -torch.log((2*torch.pi*v)**0.5)

    scaled_lml = torch.sum((p + c)/num_atoms.sqrt())
    return -scaled_lml / batch_size

def train_bnn_without_schedulers(ani_class: network_classes.ANI,
                                 train_data: torchani.data.TransformableIterable,
                                 val_data: torchani.data.TransformableIterable,
                                 AdamW,
                                 SGD,
                                 checkpoint_filename:str = "ani_model.pt",
                                 learning_curve_csv_filename:str = "learning_curve.csv",
                                 max_epochs: int = 2,
                                 early_stopping_learning_rate : float = 1.0e-9,
                                 train_mc_runs: int = 2,
                                 test_mc_runs: int = 2,
                                 kl_weight: float = 1.0,
                                 epoch: int = 0,
                                 lml: bool = False
                                 ):
    """
    Function to train BNN ANI model without schedulers
    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param AdamW: torch.optim.AdamW - instance for the AdamW optimizer
    :param SGD: torch.optim.SGD - instance for the SGD optimizer
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param train_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param test_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param kl_weight: float - weight for the KL term
    :param epoch: int - epoch number to start from
    :param lml: bool = whether to use lml loss in place of rmse
    :return: float - loss, note model is stored in the ANI class

    """
    log = logging.getLogger(__name__)

    latest_checkpoint = f"latest_{checkpoint_filename}"
    best_model_checkpoint = f"best_{checkpoint_filename}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ani_class.model.to(device)

    # Replace scheduler with this dictionary so you can keep track of the variables still
    AdamW_monitor = {}
    AdamW_monitor["last_epoch"] = epoch
    AdamW_monitor["best"] = 10000

    log.info('train data length:{}'.format(len(train_data)))

    tensorboard = torch.utils.tensorboard.SummaryWriter()

    if os.path.isfile(learning_curve_csv_filename):
        learning_curve_df = pd.read_csv(learning_curve_csv_filename)
    else:
        learning_curve_df = pd.DataFrame(columns=["Epoch", "lr", "RMSE"])

    mse = torch.nn.MSELoss(reduction="none")

    log.info("Training starting from epoch:{}".format(AdamW_monitor["last_epoch"]))
    log.info("Early stopping rate is {}".format(early_stopping_learning_rate))
    for epoch in range(AdamW_monitor["last_epoch"], max_epochs):
        AdamW_monitor["last_epoch"] = epoch
        rmse_with_mc_runs = validate_bnn(ani_class.model, val_data, AdamW_monitor["last_epoch"], test_mc_runs, kl_weight)
        log.info("Validation Loss (RMSE + KL): {} at Epoch:{}".format(rmse_with_mc_runs, AdamW_monitor["last_epoch"]))

        learning_rate = AdamW.param_groups[0]["lr"]

        learning_curve_df = learning_curve_df.append(
            {"Epoch": AdamW_monitor["last_epoch"],
             "lr": AdamW.param_groups[0]["lr"],
             "RMSE": rmse_with_mc_runs
             },
            ignore_index=True,
        )

        learning_curve_df.to_csv(learning_curve_csv_filename, index=False)

        if learning_rate < early_stopping_learning_rate:
            break

        # best checkpoint
        if rmse_with_mc_runs < AdamW_monitor["best"]:
            if ani_class.isparallel is False:
                torch.save(ani_class.model[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            else:
                torch.save(ani_class.model.module[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            AdamW_monitor["best"] = rmse_with_mc_runs

        tensorboard.add_scalar("validation_rmse",
                               rmse_with_mc_runs,
                               AdamW_monitor["last_epoch"]
                               )

        tensorboard.add_scalar(
            "best_validation_rmse",
            AdamW_monitor["best"],
            AdamW_monitor["last_epoch"]
        )

        tensorboard.add_scalar(
            "learning_rate",
            learning_rate,
            AdamW_monitor["last_epoch"]
        )

        for i, properties in tqdm.tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc="epoch {}".format(AdamW_monitor["last_epoch"]),
        ):

            species = properties["species"].to(device)
            # for auto diffrentation
            if ani_class.forces is True:
                coordinates = properties["coordinates"].to(device).float().requires_grad_(True)
            else:
                coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            batch_size = len(species)
            log.debug("Batch size: {}".format(batch_size))

            pred_energies_ = []
            kl_ = []

            for mc_run in range(train_mc_runs):
                _, pred_energies = ani_class.model((species, coordinates))
                pred_energies_.append(pred_energies)
                kl_loss = get_kl_loss(ani_class.model)
                kl_.append(kl_loss)

            predicted_energies = torch.stack(pred_energies_)

            kl = torch.mean(torch.stack(kl_), dim=0)
            scaled_kl = kl / batch_size

            if not lml:
                energy_loss = mse_e_loss(predicted_energies, true_energies, num_atoms, mse)
            else:
                energy_loss = lml_loss(predicted_energies, true_energies, num_atoms, log=log)

            # TODO: is there a better way to do this using the monte carlo runs
            if ani_class.forces is True:
                true_forces = properties['forces'].to(device).float()
                # https://aiqm.github.io/torchani/examples/nnp_training_force.html
                forces = - \
                torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

                # Now the total loss has two parts, energy loss and force loss
                force_loss = mse_f_loss(forces, true_forces, num_atoms, mse)

                loss = energy_loss + (ani_class.force_scalar * force_loss) + (kl_weight * scaled_kl)
                log.debug(f"energy loss {energy_loss} force loss {force_loss} force scaler {ani_class.force_scalar} "
                         f"kl loss {scaled_kl} kl weight {kl_weight} total loss {loss}")

            else:
                loss = energy_loss + (kl_weight * scaled_kl)
                log.debug(f"energy loss {energy_loss} kl loss {scaled_kl} kl weight {kl_weight} total loss {loss}")


            log.debug("Training Loss: {}".format(loss))
            AdamW.zero_grad()
            SGD.zero_grad()
            loss.backward()

            AdamW.step()
            SGD.step()

            # write current batch loss to TensorBoard
            tensorboard.add_scalar(
                "batch_loss", loss, AdamW_monitor["last_epoch"] * len(train_data) + i
            )
        log.info('Saving model to: {}'.format(latest_checkpoint))
        if ani_class.isparallel is False:
            torch.save(
                {
                    "nn": ani_class.model[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )
        else:
            torch.save(
                {
                    "nn": ani_class.model.module[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )

        if epoch == max_epochs - 1:
            rmse_with_mc_runs = validate_bnn(ani_class.model, val_data, AdamW_scheduler.last_epoch, test_mc_runs,
                                             kl_weight)

            log.info(
                "Validation Loss (RMSE + KL): {} at Epoch:{}".format(rmse_with_mc_runs, AdamW_scheduler.last_epoch))

    return AdamW_monitor["best"]


def train_bnn_with_schedulers(ani_class: network_classes.ANI,
                              train_data: torchani.data.TransformableIterable,
                              val_data: torchani.data.TransformableIterable,
                              AdamW,
                              SGD,
                              AdamW_scheduler,
                              SGD_scheduler,
                              checkpoint_filename: str = "ani_model.pt",
                              learning_curve_csv_filename: str = "learning_curve.csv",
                              max_epochs: int = 2,
                              early_stopping_learning_rate : float = 1.0e-9,
                              train_mc_runs: int = 2,
                              test_mc_runs: int = 2,
                              kl_weight: float = 1.0,
                              epoch: int = 0,
                              lml: bool = False,
                              ):
    """
    Function to train BNN ANI model with schedulers
    :param ani_class: network_classes.ANI - ANI network model
    :param train_data: torchani.data.TransformableIterable - training data
    :param val_data: torchani.data.TransformableIterable - validation data
    :param AdamW: torch.optim.AdamW - instance for the AdamW optimizer
    :param SGD: torch.optim.SGD - instance for the SGD optimizer
    :param AdamW_scheduler: torch.optim.lr_scheduler - instance for learn rate scheduler for the AdamW optimizer
    :param SGD_scheduler: torch.optim.lr_scheduler - instance for learn rate scheduler for the SGD optimizer
    :param checkpoint_filename:  str - checkpoint file name will have best and latest prepended
    :param learning_curve_csv_filename: str - file name for learning curve csv file to be save to
    :param max_epochs: int - maximum number of epochs
    :param early_stopping_learning_rate: float - if the learning rate reaches this low stop
    :param train_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param test_mc_runs: int - number of monte carlo runs to use to sample the posterior
    :param kl_weight: float - weight for the KL term
    :param epoch: int - epoch number to start from
    :param lml: bool = whether to use lml loss in place of rmse
    :return: float - loss, note model is stored in the ANI class
    """
    log = logging.getLogger(__name__)

    latest_checkpoint = f"latest_{checkpoint_filename}"
    best_model_checkpoint = f"best_{checkpoint_filename}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ani_class.model.to(device)

    log.info('train data length:{}'.format(len(train_data)))

    tensorboard = torch.utils.tensorboard.SummaryWriter()

    if os.path.isfile(learning_curve_csv_filename):
        learning_curve_df = pd.read_csv(learning_curve_csv_filename)
    else:
        learning_curve_df = pd.DataFrame(columns=["epoch", "lr", "loss"])

    mse = torch.nn.MSELoss(reduction="none")

    AdamW_scheduler.last_epoch = epoch

    log.info("Training starting from epoch:{}".format(AdamW_scheduler.last_epoch))
    log.info("Early stopping rate is {}".format(early_stopping_learning_rate))
    for epoch in range(AdamW_scheduler.last_epoch, max_epochs):
        AdamW_scheduler.last_epoch = epoch
        rmse_with_mc_runs = validate_bnn(ani_class.model, val_data, AdamW_scheduler.last_epoch, test_mc_runs, kl_weight)

        log.info("Validation Loss (RMSE + KL): {} at Epoch:{}".format(rmse_with_mc_runs, AdamW_scheduler.last_epoch))

        learning_rate = AdamW.param_groups[0]["lr"]

        learning_curve_df = learning_curve_df.append(
            {"epoch": AdamW_scheduler.last_epoch,
             "lr": AdamW.param_groups[0]["lr"],
             "loss": rmse_with_mc_runs
             },
            ignore_index=True,
        )

        learning_curve_df.to_csv(learning_curve_csv_filename, index=False)

        if learning_rate < early_stopping_learning_rate:
            break

        # best checkpoint
        if AdamW_scheduler.is_better(rmse_with_mc_runs, AdamW_scheduler.best):
            if ani_class.isparallel is False:
                torch.save(ani_class.model[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)
            else:
                torch.save(ani_class.model.module[ani_class._network_layer_indx].state_dict(), best_model_checkpoint)

            AdamW_scheduler.best = rmse_with_mc_runs

        # Note gradient prop is done with mc_runs equal to the largest of 1% of requests mc_runs and 2. This due to the
        # gradients being needed for the schedulers but the memory requirement being very high for a BNN over
        # lots of mc_runs.
        # rmse = validate_bnn_with_grads(ani_class.model, val_data, epoch, mc_runs=max(int(mc_runs*0.01), 2),
        #                                kl_weight=kl_weight)
        AdamW_scheduler.step(rmse_with_mc_runs)
        SGD_scheduler.step(rmse_with_mc_runs)

        tensorboard.add_scalar("validation_rmse",
                               rmse_with_mc_runs,
                               AdamW_scheduler.last_epoch
                               )

        tensorboard.add_scalar(
            "best_validation_rmse",
            AdamW_scheduler.best,
            AdamW_scheduler.last_epoch
        )

        tensorboard.add_scalar(
            "learning_rate",
            learning_rate,
            AdamW_scheduler.last_epoch
        )

        log.info("\n----- Training epoch {} -----\n".format(epoch))
        for i, properties in tqdm.tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc="epoch {}".format(AdamW_scheduler.last_epoch),
        ):

            species = properties["species"].to(device)
            # for auto diffrentation
            if ani_class.forces is True:
                coordinates = properties["coordinates"].to(device).float().requires_grad_(True)
            else:
                coordinates = properties["coordinates"].to(device).float()
            true_energies = properties["energies"].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            batch_size = len(species)
            log.debug("Batch size: {}".format(batch_size))

            pred_energies_ = []
            kl_ = []

            for mc_run in range(train_mc_runs):
                _, pred_energies = ani_class.model((species, coordinates))
                pred_energies_.append(pred_energies)
                kl_loss = get_kl_loss(ani_class.model)
                kl_.append(kl_loss)

            predicted_energies = torch.stack(pred_energies_)
            kl = torch.mean(torch.stack(kl_), dim=0)
            scaled_kl = kl / batch_size

            if not lml:
                energy_loss = mse_e_loss(predicted_energies, true_energies, num_atoms, mse)
            else:
                energy_loss = lml_loss(predicted_energies, true_energies, num_atoms, log=log)

            # TODO: is there a better way to do this using the monte carlo runs
            log.debug(f"Use forces? {ani_class.forces}")
            if ani_class.forces is True:
                true_forces = properties["forces"].to(device).float()
                # https://aiqm.github.io/torchani/examples/nnp_training_force.html
                forces = - \
                    torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

                # Now the total loss has two parts, energy loss and force loss
                force_loss = mse_f_loss(forces, true_forces, num_atoms, mse)

                loss = energy_loss + (ani_class.force_scalar * force_loss) + (kl_weight * scaled_kl)
                log.debug(f"energy loss {energy_loss} force loss {force_loss} force scaler {ani_class.force_scalar} "
                         f"kl loss {scaled_kl} kl weight {kl_weight} total loss {loss}")

            else:
                loss = energy_loss + (kl_weight * scaled_kl)
                log.debug(f"energy loss {energy_loss} kl loss {scaled_kl} kl weight {kl_weight} total loss {loss}")

            log.debug("Training Loss: {}".format(loss))
            AdamW.zero_grad()
            SGD.zero_grad()
            loss.backward()

            AdamW.step()
            SGD.step()

            # write current batch loss to TensorBoard
            tensorboard.add_scalar(
                "batch_loss", loss, AdamW_scheduler.last_epoch * len(train_data) + i
            )
        log.info('Saving model to: {}'.format(latest_checkpoint))
        if ani_class.isparallel is False:
            torch.save(
                {
                    "nn": ani_class.model[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )
        else:
            torch.save(
                {
                    "nn": ani_class.model.module[ani_class._network_layer_indx].state_dict(),
                    "AdamW": AdamW.state_dict(),
                    "SGD": SGD.state_dict(),
                    "epoch": epoch
                },
                latest_checkpoint,
            )

        if epoch == max_epochs - 1:
            rmse_with_mc_runs = validate_bnn(ani_class.model, val_data, AdamW_scheduler.last_epoch, test_mc_runs,
                                             kl_weight)

            log.info(
                "Validation Loss (RMSE + KL): {} at Epoch:{}".format(rmse_with_mc_runs, AdamW_scheduler.last_epoch))

    return AdamW_scheduler.best

