#!/usr/bin/env python
"""
This module contains code to build and modify neural potentials using the Behler Parinello concepts and smith et al's
ANI models. The library bayesian torch is used to enable the building of bayesian neural potentials
https://github.com/IntelLabs/bayesian-torch.
"""

import logging
from operator import mod, ne
from typing import Any, NoReturn, Union
import numpy as np
import torch
import torchani
import os
from bayesian_torch.models import dnn_to_bnn
import pickle
import re
from collections import OrderedDict

__title__ = os.path.basename(__file__)

#### BNN convert functions #####

def bnn_priors(reparameterization: bool = False,
               flipout: bool = False,
               prior_mu: float = 0.0,
               prior_sigma: float = 1.0,
               posterior_mu_init: float = 0.0,
               posterior_rho_init: float = -3.0,
               moped_enable: bool = True,
               moped_delta: float = 0.2):
    """
    Function to set prior parameter sets from user inputs. See https://arxiv.org/abs/1906.05323 for details of the
    implementation and the implementation is https://github.com/IntelLabs/bayesian-torch/tree/main/.

    :param reparameterization: bool - use reparameterization layers for Bayesian network
    :param flipout: bool - use flipout layers for Bayesian network
    :param prior_mu: float - prior mean
    :param prior_sigma: float - prior standard deviation
    :param posterior_mu_init: float - initialization of the posterior mean
    :param posterior_rho_init: float - initialization of the posterior variational parameter rho for representing
                                       posterior sigma through softplus function
    :param moped_enable: bool - use MOPED if used estimates the prior mean and sigma from deterministic DNN using MLE.
                                Variational posterior parameters are estimated using MLE and sigma as a fraction
                                (moped_delta) of the MLE
    :param moped_delta: float - Used to set variational parameters sigma. This is the fraction of the MLE estimate of
                                the mean used as sigma
    :return: dict - bnn_prior_parameters
    """

    log = logging.getLogger(__name__)

    if reparameterization is False and flipout is False:
        log.error("ERROR - must choose one of flipout (--flipout)  or reparameterization (--reparameterization)")
        raise RuntimeError("ERROR - user must specify one of reparameterization or flipout Bayesian layer types")

    elif reparameterization is True:
        bnn_prior_parameters = {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Reparameterization",
            "moped_enable": moped_enable,
            "moped_delta": moped_delta,
        }
    elif flipout is True:
        bnn_prior_parameters = {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Flipout",
            "moped_enable": moped_enable,
            "moped_delta": moped_delta,
        }
    else:
        log.warning("Something unforeseen has gone wrong building the Bayesian options check your inputs")
        bnn_prior_parameters = {}

    return bnn_prior_parameters


def convert_dnn_to_bnn_original(model: torch.nn.Sequential,
                                bnn_prior: dict,
                                params_prior: dict,
                                bayesian_depth: Union[int, None] = None,
                                prior_key: str = ""):
    """
    Function to convert ani to BNN to a set depth on the network
    :param model : torch.nn.Sequential - pytorch network
    :param bnn_prior: dict - output of chosen prior from bnn_priors
    :param params_prior : dict - dictionary keys are layers named by numbers as in model.named_parameters()
                                 and to sub-dictionary where keys of weight_mean, weight_std, bias_mean and bias_std
                                 set prior
    :param bayesian_depth : int - depth of layers to convert to bayesian
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
                convert_dnn_to_bnn_original(model._modules[name], bnn_prior, params_prior,
                                            bayesian_depth=bayesian_depth, prior_key=prior_key)
                prior_key = ""
            else:
                prior_key_mid = prior_key + "." + name
                convert_dnn_to_bnn_original(model._modules[name], bnn_prior, params_prior,
                                            bayesian_depth=bayesian_depth, prior_key=prior_key_mid)

        elif "Conv" in model._modules[name].__class__.__name__:
            log.debug("'Conv' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            if bayesian_depth is None:

                if params_prior is not None:
                    # Set prior to match to current distribution NOTE MOPED WILL COPY ALL PARAMETER DISTRIBUTIONS
                    bnn_prior["prior_mu"] = params_prior[local_prior_key]["weight_mean"]
                    bnn_prior["prior_sigma"] = params_prior[local_prior_key]["weight_std"]
                    bnn_prior["posterior_mu_init"] = params_prior[local_prior_key]["weight_mean"]

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_conv_layer(bnn_prior, model._modules[name]))

            elif int(name) >= bayesian_depth:

                if params_prior is not None:
                    # Set prior to match to current distribution NOTE MOPED WILL COPY ALL PARAMETER DISTRIBUTIONS
                    bnn_prior["prior_mu"] = params_prior[local_prior_key]["weight_mean"]
                    bnn_prior["prior_sigma"] = params_prior[local_prior_key]["weight_std"]

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_conv_layer(bnn_prior, model._modules[name]))

            else:
                pass

        elif "Linear" in model._modules[name].__class__.__name__:
            log.debug("'Linear' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            # None = apply to all layers
            if bayesian_depth is None:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_linear_layer(bnn_prior, model._modules[name]))

                log.info(f"AFTER BNN CONVERTER LAYER NAME: {model._modules[name].__class__.__name__}")
                log.info(f"weights\n{model._modules[name].rho_weight.data}")
                log.info(f"bias\n{model._modules[name].rho_bias.data}")

                if params_prior is not None:
                    # Set prior to match to current distribution NOTE MOPED WILL COPY ALL PARAMETER DISTRIBUTIONS
                    model._modules[name].mu_weight.data = params_prior[local_prior_key]["weight_mean"]
                    model._modules[name].rho_weight.data = params_prior[local_prior_key]["weight_rho"]
                    if bnn_layer.mu_bias is not None:
                        model._modules[name].mu_bias.data = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].rho_bias.data = params_prior[local_prior_key]["bias_rho"]

            # int = apply to the layer specified by int and all subsequent layers
            elif int(name) >= bayesian_depth:

                if params_prior is not None:
                    # Set prior to match to current distribution NOTE MOPED WILL COPY ALL PARAMETER DISTRIBUTIONS
                    log.debug("params_prior.keys() {}".format(params_prior))
                    log.debug("params_prior[{}].keys() {}".format(local_prior_key,
                                                                  params_prior[local_prior_key].keys()))
                    log.debug("params_prior[{}]['weight_mean'] {}".format(local_prior_key,
                                                                          params_prior[local_prior_key]["weight_mean"]))
                    log.debug("params_prior[{}]['weight_std'] {}".format(local_prior_key,
                                                                         params_prior[local_prior_key]["weight_std"]))
                    bnn_prior["prior_mu"] = params_prior[local_prior_key]["weight_mean"]
                    bnn_prior["prior_sigma"] = params_prior[local_prior_key]["weight_std"]

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_linear_layer(bnn_prior, model._modules[name]))


            else:
                pass

        elif "LSTM" in model._modules[name].__class__.__name__:
            log.debug("'LSTM' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            if bayesian_depth is None:

                if params_prior is not None:
                    # Set prior to match to current distribution NOTE MOPED WILL COPY ALL PARAMETER DISTRIBUTIONS
                    bnn_prior["prior_mu"] = params_prior[local_prior_key]["weight_mean"]
                    bnn_prior["prior_sigma"] = params_prior[local_prior_key]["weight_std"]

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_lstm_layer(bnn_prior, model._modules[name]))


            elif int(name) >= bayesian_depth:
                if params_prior is not None:
                    # Set prior to match to current distribution NOTE MOPED WILL COPY ALL PARAMETER DISTRIBUTIONS
                    bnn_prior["prior_mu"] = params_prior[local_prior_key]["weight_mean"]
                    bnn_prior["prior_sigma"] = params_prior[local_prior_key]["weight_std"]

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_lstm_layer(bnn_prior, model._modules[name]))

            else:
                pass
        else:
            pass


def convert_dnn_to_bnn(model: torch.nn.Sequential,
                       bnn_prior: dict,
                       params_prior: dict,
                       bayesian_depth: Union[int, None] = None,
                       set_prior_explicitly: bool = False,
                       set_rho_explicitly: bool = False,
                       prior_key: str = ""):
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
            log.debug("Prior key: {}".format(prior_key))
            if prior_key == "":
                prior_key = name
                convert_dnn_to_bnn(model._modules[name],
                                   bnn_prior,
                                   params_prior,
                                   bayesian_depth=bayesian_depth,
                                   set_prior_explicitly=set_prior_explicitly,
                                   set_rho_explicitly=set_rho_explicitly,
                                   prior_key=prior_key)
                prior_key = ""
            else:
                prior_key_mid = prior_key + "." + name
                convert_dnn_to_bnn(model._modules[name],
                                   bnn_prior,
                                   params_prior,
                                   bayesian_depth=bayesian_depth,
                                   set_prior_explicitly=set_prior_explicitly,
                                   set_rho_explicitly=set_rho_explicitly,
                                   prior_key=prior_key_mid)

        elif re.search(r"^Conv", model._modules[name].__class__.__name__):
            log.debug("'Conv' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            if bayesian_depth is None:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_conv_layer(bnn_prior, model._modules[name]))

                if params_prior is not None:
                    # Here we set the prior to match to the values we specify. Ideally this keeps the model in
                    # the same ball park.
                    if set_prior_explicitly is True:
                        model._modules[name].prior_weight_mu = params_prior[local_prior_key]["weight_mean"]
                        model._modules[name].prior_weight_sigma = params_prior[local_prior_key]["weight_std"]
                        model._modules[name].prior_bias_mu = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].prior_bias_sigma = params_prior[local_prior_key]["bias_std"]

                    # TODO: It is not exactly clear in the code if all of these should be set to these values we should
                    #  check
                    # Here we set the weights of the parameter distributions to match to those that we pass
                    # In this case that means that the mean weight is set to the mean value and the variational
                    # parameter rho which sets the variance of the Gaussian distribution is set such that when
                    # passed through a soft plus function ln(1 + e^(weight_rho)) we get the std (weight_std)

                    # 1. Set the mean of the weights distribution
                    model._modules[name].mu_weight.data.copy_(params_prior[local_prior_key]["weight_mean"])

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                      ["weight_mean"],
                                                                                      bnn_prior["moped_delta"]))
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        model._modules[name].mu_bias.data.copy_(params_prior[local_prior_key]["bias_mean"])
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                        ["bias_mean"],
                                                                                        bnn_prior["moped_delta"]))

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            elif int(name) >= bayesian_depth:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_conv_layer(bnn_prior, model._modules[name]))

                if params_prior is not None:
                    # Here we set the prior to match to the values we specify. Ideally this keeps the model in
                    # the same ball park.
                    if set_prior_explicitly is True:
                        model._modules[name].prior_weight_mu = params_prior[local_prior_key]["weight_mean"]
                        model._modules[name].prior_weight_sigma = params_prior[local_prior_key]["weight_std"]
                        model._modules[name].prior_bias_mu = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].prior_bias_sigma = params_prior[local_prior_key]["bias_std"]

                    # TODO: It is not exactly clear in the code if all of these should be set to these values we should
                    #  check
                    # Here we set the weights of the parameter distributions to match to those that we pass
                    # In this case that means that the mean weight is set to the mean value and the variational
                    # parameter rho which sets the variance of the Gaussian distribution is set such that when
                    # passed through a soft plus function ln(1 + e^(weight_rho)) we get the std (weight_std)

                    # 1. Set the mean of the weights distribution
                    model._modules[name].mu_weight.data.copy_(params_prior[local_prior_key]["weight_mean"])

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                      ["weight_mean"],
                                                                                      bnn_prior["moped_delta"]))
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        model._modules[name].mu_bias.data.copy_(params_prior[local_prior_key]["bias_mean"])
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                        ["bias_mean"],
                                                                                        bnn_prior["moped_delta"]))

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            else:
                pass

        elif re.search(r"^Linear",  model._modules[name].__class__.__name__):
            log.debug("'Linear' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            # None = apply to all layers
            if bayesian_depth is None:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_linear_layer(bnn_prior, model._modules[name]))

                # log.info(f"AFTER BNN CONVERTER LAYER NAME: {model._modules[name].__class__.__name__}")
                # log.info(f"weights\n{model._modules[name].rho_weight.data}")
                # log.info(f"bias\n{model._modules[name].rho_bias.data}")

                # NOTES:
                # rho_weights seem to set the posterior variance if we set them explicitly then the KL is very close
                # The problem is when I set it somewhere we seem to loose the ability to modify the tensor. I suspect
                # we are loosing a gradient trace somewhere. Other option is that perhaps they need re-registering?
                #
                # The prior seems to be what we should set to be close to ANI ensemble. This is now set as below. None
                # of the prior terms should need gradients and appear not to need them.
                # NOTE: Bayesian torch sets only the mu and rho terms not the prior terms

                if params_prior is not None:
                    # Here we set the prior to match to the values we specify. Ideally this keeps the model in
                    # the same ball park.
                    if set_prior_explicitly is True:
                        model._modules[name].prior_weight_mu = params_prior[local_prior_key]["weight_mean"]
                        model._modules[name].prior_weight_sigma = params_prior[local_prior_key]["weight_std"]
                        model._modules[name].prior_bias_mu = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].prior_bias_sigma = params_prior[local_prior_key]["bias_std"]

                    # Here we set the weights of the parameter distributions to match to those that we pass
                    # In this case that means that the mean weight is set to the mean value and the variational
                    # parameter rho which sets the variance of the Gaussian distribution is set such that when
                    # passed through a soft plus function ln(1 + e^(weight_rho)) we get the std (weight_std)

                    # 1. Set the mean of the weights distribution
                    model._modules[name].mu_weight.data.copy_(params_prior[local_prior_key]["weight_mean"])

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in note if we pass them in they should be transformed as rho values from
                    # standard deviations.
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                 ["weight_mean"],
                                                                                 bnn_prior["moped_delta"]))
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        model._modules[name].mu_bias.data.copy_(params_prior[local_prior_key]["bias_mean"])
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                     ["bias_mean"],
                                                                                     bnn_prior["moped_delta"]))

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in note if we pass them in they should be transformed as rho values from
                        # standard deviations.
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            # int = apply to the layer specified by int and all subsequent layers
            elif int(name) >= bayesian_depth:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_linear_layer(bnn_prior, model._modules[name]))

                if params_prior is not None:
                    # Here we set the prior to match to the values we specify. Ideally this keeps the model in
                    # the same ball park.
                    if set_prior_explicitly is True:
                        model._modules[name].prior_weight_mu = params_prior[local_prior_key]["weight_mean"]
                        model._modules[name].prior_weight_sigma = params_prior[local_prior_key]["weight_std"]
                        model._modules[name].prior_bias_mu = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].prior_bias_sigma = params_prior[local_prior_key]["bias_std"]

                    # TODO: It is not exactly clear in the code if all of these should be set to these values we should
                    #  check
                    # Here we set the weights of the parameter distributions to match to those that we pass
                    # In this case that means that the mean weight is set to the mean value and the variational
                    # parameter rho which sets the variance of the Gaussian distribution is set such that when
                    # passed through a soft plus function ln(1 + e^(weight_rho)) we get the std (weight_std)

                    # 1. Set the mean of the weights distribution
                    model._modules[name].mu_weight.data.copy_(params_prior[local_prior_key]["weight_mean"])

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                      ["weight_mean"],
                                                                                      bnn_prior["moped_delta"]))
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        model._modules[name].mu_bias.data.copy_(params_prior[local_prior_key]["bias_mean"])
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                        ["bias_mean"],
                                                                                        bnn_prior["moped_delta"]))

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            else:
                pass

        elif re.search(r"^LSTM", model._modules[name].__class__.__name__):
            log.debug("'LSTM' in model._modules[name].__class__.__name__: True")
            local_prior_key = prior_key + "." + name
            log.debug("Prior key {}".format(local_prior_key))

            if bayesian_depth is None:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_lstm_layer(bnn_prior, model._modules[name]))

                if params_prior is not None:
                    # Here we set the prior to match to the values we specify. Ideally this keeps the model in
                    # the same ball park.
                    if set_prior_explicitly is True:
                        model._modules[name].prior_weight_mu = params_prior[local_prior_key]["weight_mean"]
                        model._modules[name].prior_weight_sigma = params_prior[local_prior_key]["weight_std"]
                        model._modules[name].prior_bias_mu = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].prior_bias_sigma = params_prior[local_prior_key]["bias_std"]

                    # TODO: It is not exactly clear in the code if all of these should be set to these values we should
                    #  check
                    # Here we set the weights of the parameter distributions to match to those that we pass
                    # In this case that means that the mean weight is set to the mean value and the variational
                    # parameter rho which sets the variance of the Gaussian distribution is set such that when
                    # passed through a soft plus function ln(1 + e^(weight_rho)) we get the std (weight_std)

                    # 1. Set the mean of the weights distribution
                    model._modules[name].mu_weight.data.copy_(params_prior[local_prior_key]["weight_mean"])

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                      ["weight_mean"],
                                                                                      bnn_prior["moped_delta"]))
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        model._modules[name].mu_bias.data.copy_(params_prior[local_prior_key]["bias_mean"])
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                        ["bias_mean"],
                                                                                        bnn_prior["moped_delta"]))

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])


            elif int(name) >= bayesian_depth:

                setattr(
                    model,
                    name,
                    dnn_to_bnn.bnn_lstm_layer(bnn_prior, model._modules[name]))

                if params_prior is not None:
                    # Here we set the prior to match to the values we specify. Ideally this keeps the model in
                    # the same ball park.
                    if set_prior_explicitly is True:
                        model._modules[name].prior_weight_mu = params_prior[local_prior_key]["weight_mean"]
                        model._modules[name].prior_weight_sigma = params_prior[local_prior_key]["weight_std"]
                        model._modules[name].prior_bias_mu = params_prior[local_prior_key]["bias_mean"]
                        model._modules[name].prior_bias_sigma = params_prior[local_prior_key]["bias_std"]

                    # TODO: It is not exactly clear in the code if all of these should be set to these values we should
                    #  check
                    # Here we set the weights of the parameter distributions to match to those that we pass
                    # In this case that means that the mean weight is set to the mean value and the variational
                    # parameter rho which sets the variance of the Gaussian distribution is set such that when
                    # passed through a soft plus function ln(1 + e^(weight_rho)) we get the std (weight_std)

                    # 1. Set the mean of the weights distribution
                    model._modules[name].mu_weight.data.copy_(params_prior[local_prior_key]["weight_mean"])

                    # 2. Set the variance of the weights distribution either (False) as a fraction of the mean or (True)
                    # as the values we pass in
                    if set_rho_explicitly is False:
                        # NOTE get_rho multiplies the mean with delta to get the sigma value
                        model._modules[name].rho_weight.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                      ["weight_mean"],
                                                                                      bnn_prior["moped_delta"]))
                    elif set_rho_explicitly is True:
                        model._modules[name].rho_weight.data.copy_(params_prior[local_prior_key]["weight_rho"])

                    # If there are biases used in the model set them to the values we have
                    if model._modules[name].mu_bias is not None:

                        # 3. Set the mean of the bias distribution
                        model._modules[name].mu_bias.data.copy_(params_prior[local_prior_key]["bias_mean"])
                        if set_rho_explicitly is False:
                            # NOTE get_rho multiplies the mean with delta to get the sigma value
                            model._modules[name].rho_bias.data.copy_(dnn_to_bnn.get_rho(params_prior[local_prior_key]
                                                                                        ["bias_mean"],
                                                                                        bnn_prior["moped_delta"]))

                        # 4. Set the variance of the bias distribution either (False) as a fraction of the mean or (True)
                        # as the values we pass in
                        elif set_rho_explicitly is True:
                            model._modules[name].rho_bias.data.copy_(params_prior[local_prior_key]["bias_rho"])

            else:
                pass
        else:
            pass

def update_bnn_variance(model: torch.nn.Sequential,
                       bnn_prior: dict,
                       params_prior: dict,
                       bayesian_depth: Union[int, None] = None,
                       set_prior_explicitly: bool = False,
                       set_rho_explicitly: bool = False,
                       prior_key: str = ""):
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

                            model._modules[name].rho_bias.data.copy_(
                                dnn_to_bnn.get_rho(
                                    model._modules[name].mu_bias.data,
                                    bnn_prior["moped_delta"]
                                )
                            )

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

                        model._modules[name].rho_bias.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_bias.data,
                                bnn_prior["moped_delta"]
                            )
                        )

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

                        model._modules[name].rho_bias.data.copy_(
                            dnn_to_bnn.get_rho(
                                model._modules[name].mu_bias.data,
                                bnn_prior["moped_delta"]
                            )
                        )

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

def is_bayesian_network(model: torch.nn.Sequential,
                        prior_key: str = "") -> bool:
    """
    Function to check if a model is bayesian.

    This function sequentially checks the layer names looking for bayesian torch layer naming.
    The function sets the property "bayesian" of the class when called. Note it makes no distinction
    between fully bayesian i.e. all layers are bayesian and some layers being bayesian.

    :param model: torch.nn.Sequential - model network
    :param prior_key: str - SHOULD NOT BE SET EXPLICITLY TO ANYTHING OTHER THAN ''
                            IT IS USED IN THE RECURSIVE CALLS BY THIS FUNCTION
    :return: bool
    """

    log = logging.getLogger(__name__)

    bayesian = False

    for name, value in list(model._modules.items()):

        log.debug("Is Bayesian: {} {}".format(name, value))

        if model._modules[name]._modules:
            log.debug("model._modules[name]._modules: True")
            log.debug("Name: {}".format(name))
            if prior_key == "":
                prior_key = name
                bayesian = is_bayesian_network(model._modules[name],
                                   prior_key=prior_key)
                prior_key = ""
            else:
                prior_key_mid = prior_key + "." + name
                bayesian = is_bayesian_network(model._modules[name],
                                prior_key=prior_key_mid)

        elif "Reparameterization" in model._modules[name].__class__.__name__:
            bayesian = True
            log.debug("Reparameterization layer used in model")
            break

        elif "Flipout" in model._modules[name].__class__.__name__:
            bayesian = True
            log.debug("Flipout layer used in model")
            break

    return bayesian


#### Classes #####

class TorchNetwork(object):
    """
    Base class for Pytorch based networks. This base class contains most of the functionality for loading a general
    pytorch network and converting it to a Bayesain network using Bayesian torch.
    """

    def __init__(self,
                 model: torch.nn.Sequential = None,
                 bnn_prior: dict = None,
                 params_prior: dict = None,
                 bayesian_depth: int = None,
                 name: str = None,
                 forces: bool = False,
                 force_scalar: float = 1.0,
                 **kwargs):
        """
        Initialize the class
        :param model: torch.nn.Sequential - Pytorch model
        :param bnn_prior: dict - if converting to a Bayesian network this is the prior specification see
                                 bayesian torch documnetation for details
        :param params_prior: dict - dictionary keys are layers named by numbers as in model.named_parameters()
                                    and to sub-dictionary where keys of weight_mean, weight_std, bias_mean and
                                    bias_std set prior
        :param bayesian_depth: int - depth of layers to convert to bayesian note this runs from the last to the first
        :param name: str - descriptive name for the model so it can be understood what it is later on
        :param forces: bool - use forces in the training of the potential
        :param force_scalar: float - weight for the forces terms
        :param kwargs: dict - other keyword arguments
        """
        self.model = model
        self.bayesian = False
        self.reparameterization = False
        self.flipout = False
        self.name = name
        self.bnn_prior = bnn_prior
        self.params_prior = params_prior
        self.bayesian_depth = bayesian_depth
        self.pretrained = None
        self.data_parallel = False
        self.distrbuted_data_parallel = False
        self.forces = forces
        self.force_scalar = force_scalar
        self.kwargs = kwargs

    def __eq__(self, other):
        """
        Function to check for equivalent classes
        :param other: TorchNetwork - The class to compare to
        :return: bool
        """

        if (isinstance(other, TorchNetwork)):
            return self.name == other.name and self.model == other.model and self.bayesian == other.bayesian
        else:
            return False

    @property
    def isbayesian(self) -> bool:
        """
        Property of the class whether the network is a Bayesian network or not
        :return: bool
        """
        self.bayesian, _, _ = self.is_bayesian_network(model=self.model)
        return self.bayesian

    @property
    def uses_reparameterization(self) -> bool:
        """
        Property of the class whether the network uses reparameterization layers or not
        :return: bool
        """
        _, self.reparameterization, _ = self.is_bayesian_network(model=self.model)
        return self.reparameterization

    @property
    def uses_flipout(self) -> bool:
        """
        Property of the class whether the network uses flipout layers or not
        :return: bool
        """
        _, _, self.flipout = self.is_bayesian_network(model=self.model)
        return self.flipout

    @property
    def uses_forces(self) -> bool:
        """
        Property of the class whether the network uses forces (-ve derivative of energies wrt coordinates) or not
        :return: bool
        """
        return self.forces

    @property
    def isparallel(self):
        """

        :return:
        """
        return True if self.data_parallel is True or self.distrbuted_data_parallel is True else False

    def is_bayesian_network(self,
                            model: torch.nn.Sequential,
                            prior_key: str = "") -> (bool, bool, bool):
        """
        Function to check if a model is bayesian and which layer types used.

        This function sequentially checks the layer names looking for bayesian torch layer naming.
        The function sets the property "bayesian" of the class when called. Note it makes no distrinction
        between fully bayesian i.e. all layers are bayesian and some layers being bayesian.

        :param model: torch.nn.Sequential - model network
        :param prior_key: str - SHOULD NOT BE SET EXPLICITLY TO ANYTHING OTHER THAN ''
                                IT IS USED IN THE RECURSIVE CALLS BY THIS FUNCTION

        :return: (bool, bool, bool) - Bayesian?, Reparameterization layers used? Flipout layers used?
        """

        log = logging.getLogger(__name__)

        bayesian = False
        reparameterization = False
        flipout = False

        for name, value in list(model._modules.items()):

            log.debug("Is Bayesian: {} {}".format(name, value))

            if model._modules[name]._modules:
                log.debug("model._modules[name]._modules: True")
                log.debug("Name: {}".format(name))
                if prior_key == "":
                    prior_key = name
                    bayesian = is_bayesian_network(model._modules[name],
                                                   prior_key=prior_key)
                    prior_key = ""
                else:
                    prior_key_mid = prior_key + "." + name
                    bayesian = is_bayesian_network(model._modules[name],
                                                   prior_key=prior_key_mid)

            elif "Reparameterization" in model._modules[name].__class__.__name__:
                bayesian = True
                reparameterization = True
                log.info("Reparameterization layer used in model")
                break

            elif "Flipout" in model._modules[name].__class__.__name__:
                bayesian = True
                flipout = True
                log.info("Flipout layer used in model")
                break

        return bayesian, reparameterization, flipout

    def transfer_dnn_to_bnn(self,
                            bnn_prior: dict,
                            params_prior: dict,
                            bayesian_depth: int = None,
                            set_rho_explicitly: bool = True,
                            set_prior_explicitly: bool = True):
        """
        Function to convert ani to BNN to a set depth on the network
        :param bnn_prior: dict - output of chosen prior from bnn_priors
        :param params_prior : dict - dictionary keys are layers named by numbers as in model.named_parameters()
                              and to sub-dictionary where keys of weight_mean, weight_std, bias_mean and bias_std set
                              prior
        :param bayesian_depth : int - depth of layers to convert to bayesian
        :param set_rho_explicitly: bool - if params prior is pass use it to set the rho array explicitly
        :param set_prior_explicitly: bool - if params prior is pass use it to set the prior arrays explicitly
        return None
        """

        log = logging.getLogger(__name__)

        self.bnn_prior = bnn_prior
        self.params_prior = params_prior

        if self.isbayesian is False:
            log.info("Converting model DNN to BNN")
            convert_dnn_to_bnn(model=self.model,
                               bnn_prior=bnn_prior,
                               params_prior=params_prior,
                               bayesian_depth=bayesian_depth,
                               set_rho_explicitly=set_rho_explicitly,
                               set_prior_explicitly=set_prior_explicitly,
                               prior_key=""
                               )
            log.info("Model DNN converted to BNN")
            self.bayesian = True
        else:
            log.info("Model is already Bayesian")

    def run_in_data_parallel(self):
        """
        Function to wrap model module in data parallel to run on multiple GPUs.
        This also sets the data_parallel flag as True so it can be checked easily.

        This can be awkward as it wraps the function hence it is harder to access generically the layers be careful.

        :return: None
        """

        self.model = torch.nn.DataParallel(self.model)
        self.data_parallel = True

    def run_in_distrbuted_data_parallel(self):
        """
        Function to wrap model module in data parallel to run on multiple GPUs.
        This also sets the data_parallel flag as True so it can be checked easily.

        This can be awkward as it wraps the function hence it is harder to access generically the layers be careful.

        :return: None
        """

        self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.distrbuted_data_parallel = True

    def load_pretrained_on_to_model(self,
                                    pretrained: str):
        """
        Function to load a pretrained model on to the equivalent initialized architecture
        :param pretrained: str - file and path to load weights from
        :return: None
        """
        log = logging.getLogger(__name__)

        self.pretrained = pretrained

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is None:
            log.error("ERROR - must initialize model first with equivalent architecture")

        if self.isbayesian is True:
            log.info("Current model is Bayesian")
        else:
            log.info("Current model is not Bayesian")

        log.info(f"Loading pretrained weights from {self.pretrained} on to current model")

        try:
            # Load weights and bias on the default network
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))
            self.model.eval()
        except KeyError as kerr:
            log.warning("\nERROR - can not load model key error. Likely you are trying to load a Bayesian or"
                        " non Bayesian architecture on to the opposite scaffold please check.")
            raise kerr

    def save(self,
             name: Union[str, None] = None):
        """
        Save and store the class
        :param name: str - descrptive name to understand what this instance was for
        :return: None
        """

        if name is None:
            if self.name is None:
                name = self.__class__.__name__
            else:
                name = self.name

        with open(f"{'_'.join(name.split())}.pkl", "wb") as fout:
            pickle.dump(self.__dict__, fout)

    def load(self,
             name: Union[str, None] = None):
        """
        read and load the class
        :param name: str - descriptive name used for equivalence checking
        :return: None
        """

        if name is None:
            if self.name is None:
                name = self.__class__.__name__
            else:
                name = self.name

        with open(f"{name}.pkl", "rb") as fin:
            self.__dict__ = pickle.load(fin)


class ANI(TorchNetwork):
    """
    ANI neural potential class for the Pytorch based networks. Inherits from the TorchNetwork class and adds
    specific functions for ANI styled neural potentials.
    """

    def __init__(self,
                 model: torch.nn.Sequential = None,
                 bnn_prior: dict = None,
                 params_prior: dict = None,
                 bayesian_depth: int = None,
                 name: str = None,
                 forces: bool = False,
                 force_scalar: float = 1.0,
                 self_energies: list = None,
                 **kwargs):
        """
        Initializes the class and all variables used to store information. Some of these may be redundant but are there
        for checking later on if needed. All arguments are used in the base class initialization
        :param model: torch.nn.Sequential - Pytorch model
        :param bnn_prior: dict - if converting to a Bayesian network this is the prior specification see
                                 bayesian torch documnetation for details
        :param params_prior: dict - dictionary keys are layers named by numbers as in model.named_parameters()
                                    and to sub-dictionary where keys of weight_mean, weight_std, bias_mean and
                                    bias_std set prior
        :param bayesian_depth: int - depth of layers to convert to bayesian note this runs from the last to the first
        :param name: str - descriptive name for the model so it can be understood what it is later on
        :param forces: bool - use forces in the training of the potential
        :param force_scalar: float - weight for the forces terms
        :apram self_energies: list - self energies for atoms
        :param kwargs: dict - other keyword arguments
        """
        super().__init__()
        self.default_ani = False
        self.networks = None
        self.radial_cutoff = None
        self.theta_max_rad = None
        self.angular_cutoff = None
        self.etar = None
        self.etaa = None
        self.zeta = None
        self.radial_steps = None
        self.angular_radial_steps = None
        self.theta_steps = None
        self.species_order = None
        self.device = None
        self.aev_computer = None
        self.use_cuaev = False
        self.num_species = None
        self.aev_dim = None
        self.h_network = None
        self.c_network = None
        self.n_network = None
        self.o_network = None
        self.self_energies = self_energies
        self.energy_shifter = torchani.utils.EnergyShifter(self_energies)
        self.sgd_ani_scheduler = None
        self.adamw_ani_scheduler = None
        self.completed_model = None
        self.ensemble = 1
        # An internal index for which layer to load pretrained weights to. If we add utility layers to convert
        # symbols or periodic table numbers to network number this might change when we 'complete' the network.
        self._network_layer_indx = 1
        self.model = model
        self.bayesian = False
        self.reparameterization = False
        self.flipout = False
        self.name = name
        self.bnn_prior = bnn_prior
        self.params_prior = params_prior
        self.bayesian_depth = bayesian_depth
        self.pretrained = None
        self.data_parallel = False
        self.forces = forces
        self.force_scalar = force_scalar
        self.kwargs = kwargs

    def __eq__(self, other):
        """
        Function to check for equivalent classes
        :param other: The class ANI to compare
        :return: bool
        """
        if (isinstance(other, ANI)):
            return self.name == other.name and self.model == other.model and self.bayesian == other.bayesian
        else:
            return False

    @property
    def iscomplete(self) -> bool:
        """
        Function to check if the model is complete i.e. has energy shifter attached to the trained network
        :return: bool
        """

        if self.completed_model is None:
            return False
        else:
            return True

    @property
    def is_multimodel_ensemble(self) -> bool:
        """
        Function to check if the model is a multimodel ensemble
        :return: bool
        """
        if self.ensemble == 1:
            return False
        else:
            return True


    def build_ani_dnn_model(self,
                        radial_cutoff: float = 5.2000e+00,
                        theta_max_rad: float = 3.33794218e+00,
                        angular_cutoff: float = 3.5000e+00,
                        etar: float = 1.6000000e+01,
                        etaa: float = 8.000000,
                        zeta: float = 3.2000000e+01,
                        radial_steps: int = 16,
                        angular_radial_steps: int = 4,
                        theta_steps: int = 8,
                        species_order: tuple = ("H", "C", "N", "O"),
                        networks: tuple = None,
                        ensemble: int = 1,
                        use_cuaev: bool = False,
                        no_species_converter: bool = False
                        ) -> None:
        """
        Function build an ANI model all in one call. If no networks are passed in default ANI networks for
        hydrogen, carbon, nitrogen and oxygen are used. If you want to give your own networks then the species order
        and networks must be given in the same order (THIS CANNOT BE CHECKED INTERNALLY) and a tuple of
        torch.nn.sequential networks should be pass in to networks.

        :param radial_cutoff: float - cutoff radius for the radial symmetry function elements
        :param theta_max_rad: float - theta max in radians
        :param angular_cutoff: float - radial cutoff for the angular features in distance units
        :param etar: float - eta the gaussians spread for the radial terms in the atomic environment vec
        :param etaa: float - eta the gaussians spread for the angle terms in the atomic environment vec
        :param zeta: float - zeta width of the angular gaussians
        :param radial_steps: int - number of radial steps
        :param angular_radial_steps: int - number of radial steps in the angle term
        :param theta_steps: int - number of angle steps
        :param species_order: list - atomic symbol list of the order of the species
        :param networks: tuple - tuple of torch.nn.Sequential one for each species
        :param ensemble: int - build an ensemble of n models
        :return: None
        """

        log = logging.getLogger(__name__)

        self.ensemble = ensemble

        if use_cuaev is True:
            self.use_cuaev = True

        if no_species_converter is False:
            spec_conv = torchani.nn.SpeciesConverter(species_order)
        else:
            spec_conv = None
            log.warning("User reuqested no species converter, species are assumed to be given by the network number "
                        "rather than atomic numbers")

        aev_computer = self.get_aev_computer(radial_cutoff,
                                            theta_max_rad,
                                            angular_cutoff,
                                            etar,
                                            etaa,
                                            zeta,
                                            radial_steps,
                                            angular_radial_steps,
                                            theta_steps,
                                            species_order,
                                            use_cuaev=use_cuaev)


        if networks is None:
            h_network, c_network, n_network, o_network = self.get_ani_network()
            networks = [h_network, c_network, n_network, o_network]
            self.default_ani = True
        else:
            log.info("Using user defined list of networks. NOTE MUST BE THE SAME ORDER AS SPECIES ORDER.\n"
                     "Species order: {}\nNetworks:\n{}\n".format(species_order, networks))
            self.default_ani = False
            if isinstance(networks, OrderedDict):
                networks = self.get_user_defined_networks(networks) 

        self.networks = networks

        self.build(aev_computer, networks, species_converter=spec_conv, ensemble=None)


    def get_aev_computer(self,
                         radial_cutoff: float = 5.2000e+00,
                         theta_max_rad: float = 3.33794218e+00,
                         angular_cutoff: float = 3.5000e+00,
                         etar: float = 1.6000000e+01,
                         etaa: float = 8.000000,
                         zeta: float = 3.2000000e+01,
                         radial_steps: int = 16,
                         angular_radial_steps: int = 4,
                         theta_steps: int = 8,
                         species_order: tuple = ("H", "C", "N", "O"),
                         use_cuaev: bool = False
                         ) -> torchani.AEVComputer:
        """
        Function to the return AEV computer that takes coordinates as input and outputs aevs
        :param radial_cutoff: float - cutoff radius for the radial symmetry function elements
        :param theta_max_rad: float - theta max in radians
        :param angular_cutoff: float - radial cutoff for the angular features in distance units
        :param etar: float - eta the gaussians spread for the radial terms in the atomic environment vec
        :param etaa: float - eta the gaussians spread for the angle terms in the atomic environment vec
        :param zeta: float - zeta width of the angular gaussians
        :param radial_steps: int - number of radial steps
        :param angular_radial_steps: int - number of radial steps in the angle term
        :param theta_steps: int - number of angle steps
        :param species_order: list - atomic symbol list of the order of the species
        :param use_cuaev: bool - use gpu to make the aev
        :return: torchani.AEVComputer
        """
        log = logging.getLogger(__name__)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store in the class so things can be checked and tracked down easily
        self.radial_cutoff = radial_cutoff
        self.theta_max_rad = theta_max_rad
        self.angular_cutoff = angular_cutoff
        self.etar = etar
        self.etaa = etaa
        self.zeta = zeta
        self.radial_steps = radial_steps
        self.angular_radial_steps = angular_radial_steps
        self.theta_steps = theta_steps
        self.species_order = species_order
        self.device = device

        # Map the options to the ANI computer featurization options
        Rcr = radial_cutoff
        Rca = angular_cutoff
        theta_max = theta_max_rad
        EtaR = torch.tensor(etar, device=device)
        ShfR = torch.tensor(
            np.linspace(
                9.0000000e-01,
                Rcr,
                radial_steps,
                endpoint=False,
                dtype=np.float32
            ),
            device=device,
        )

        EtaA = torch.tensor([etaa], device=device)
        Zeta = torch.tensor(zeta, device=device)
        ShfA = torch.tensor(
            np.linspace(
                9.0000000e-01,
                Rca,
                angular_radial_steps,
                endpoint=False,
                dtype=np.float32,
            ),
            device=device,
        )

        ShfZ = torch.tensor(
            np.linspace(
                1.9634954e-01,
                theta_max,
                theta_steps,
                endpoint=False,
                dtype=np.float32,
            ),
            device=device,
        )

        log.info('Rcr:{}'.format(Rcr))
        log.info('Rca:{}'.format(Rca))
        log.info('EtaR:{}'.format(EtaR))
        log.info('ShfR:{}'.format(ShfR))
        log.info('Zeta:{}'.format(Zeta))
        log.info('ShfZ:{}'.format(ShfZ))
        log.info('EtaA:{}'.format(EtaA))
        log.info('ShfA:{}'.format(ShfA))
        log.info('species_order:{}'.format(species_order))

        num_species = len(species_order)

        if use_cuaev is not True and self.use_cuaev is not True:
            log.info("Not using cuaev")
            aev_computer = torchani.AEVComputer(Rcr,
                                                Rca,
                                                EtaR,
                                                ShfR,
                                                EtaA,
                                                Zeta,
                                                ShfA,
                                                ShfZ,
                                                num_species)
        else:
            self.use_cuaev = True
            log.info("Using cuaev")
            aev_computer = torchani.AEVComputer(Rcr,
                                                Rca,
                                                EtaR,
                                                ShfR,
                                                EtaA,
                                                Zeta,
                                                ShfA,
                                                ShfZ,
                                                num_species,
                                                use_cuda_extension=True)


        self.aev_computer = aev_computer
        self.num_species = num_species
        self.aev_dim = aev_computer.aev_length

        return aev_computer

    def init_params(self, nn: torchani.ANIModel):
        """
        Function to initialize model parameters
        :param nn: torchani.ANIModel - ANI model
        :return: None
        """

        log = logging.getLogger(__name__)

        if isinstance(nn, torch.nn.Linear) and self.ensemble == 1:
            torch.nn.init.kaiming_normal_(nn.weight, a=1.0)
            torch.nn.init.zeros_(nn.bias)
        elif isinstance(nn, torch.nn.Linear) and self.ensemble > 1:
            a = np.random.random_sample()

            if round(a) == 1:
                nonlinearity = "relu"
            else:
                nonlinearity = "leaky_relu"

            log.debug(f"Initializing with randomized a and nonlinearity. a: {a} nonlinearity {nonlinearity}")
            torch.nn.init.kaiming_normal_(nn.weight, a=a, nonlinearity=nonlinearity)
            torch.nn.init.zeros_(nn.bias)

    def get_layers(self, definition: list):
        """
        Put together the layers from a list of lists
        :param definition: list - list to define the layer
        :return: layers
        """

        fc = torch.nn.Linear(definition[0], definition[1])

        if definition[2].isalpha():
            if definition[2].lower() == "celu":
                ac = torch.nn.CELU(*definition[3:])
        
        return fc, ac

    def get_user_defined_networks(self, networks: OrderedDict, use_ani_networks_for_ani_atoms: bool=False) -> tuple:
        """
        Automatically build from a user description a neural network for a given element
        :param networks: ordered dict - dictionary of element keys and lists of list, each sub list defines 
                                input number of neurons, output number of neurons activation fx 
                                and activation fx parameters for example:
                                {
                                    "H": [[100, 50, celu, 0.1], [50, 25, celu, 0.1], [25, 1, celu, 0.1]]
                                }
        :param use_ani_networks_for_ani_atoms: bool - use ani networks for H, C, N and O or not if they are used 
                                                      it is assumed they are the first elements
        :return tuple of torch.nn.Sequential
        """

        log = logging.getLogger(__name__)

        # output variable
        model_pots = []

        if use_ani_networks_for_ani_atoms is True:
            h_network, c_network, n_network, o_network = self.get_ani_network()
            model_pots.append(h_network)
            model_pots.append(c_network)
            model_pots.append(n_network)
            model_pots.append(o_network)
        
        for k, v in networks.items():
            log.info("Building network for {}".format(k))
            network = []
            for ent in v:
                fc, ac = self.get_layers(ent)
                network.append(fc)
                network.append(ac)
            ml_pot = torch.nn.Sequential(*network)
            log.info("Network {}:\n{}".format(k, network))
            model_pots.append(ml_pot)
        
        return model_pots

    def get_ani_network(self) -> (torch.nn.Sequential, torch.nn.Sequential, torch.nn.Sequential,
                torch.nn.Sequential):
        """
        Function to get ANI network architectures
        :returns: (torch.nn.Sequential, torch.nn.Sequential, torch.nn.Sequential,
                torch.nn.Sequential) - ANI network architectures tuple of torch.nn.Sequential
        """

        log = logging.getLogger(__name__)

        log.info('Creating ANI Network for net version: 1')
        h_network = torch.nn.Sequential(
            torch.nn.Linear(self.aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        c_network = torch.nn.Sequential(
            torch.nn.Linear(self.aev_dim, 144),
            torch.nn.CELU(0.1),
            torch.nn.Linear(144, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        n_network = torch.nn.Sequential(
            torch.nn.Linear(self.aev_dim, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        o_network = torch.nn.Sequential(
            torch.nn.Linear(self.aev_dim, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        self.h_network = h_network
        self.c_network = c_network
        self.n_network = n_network
        self.o_network = o_network

        return h_network, c_network, n_network, o_network

    def build(self,
              aev_computer: torchani.AEVComputer,
              networks: list,
              species_converter: Union[torchani.nn.SpeciesConverter, None] = None,
              ensemble: Union[int, None] = None
              ):
        """
        Function to build ANI network model
        :param aev_computer: torchani.AEVComputer - length of the atomic env vector
        :param networks:  list of torch.nn.Sequential - list of neural network architectures
        :param ensemble: int - number of models to train concurrently with different initialization to make an ensemble
        :return: None
        """

        log = logging.getLogger(__name__)

        if ensemble is None:
            ensemble = self.ensemble
        else:
            self.ensemble = ensemble

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nn_ensemble = []
        for ensem in range(ensemble):
            nn = torchani.ANIModel(networks)
            nn.apply(self.init_params)
            nn_ensemble.append(nn)

        nn = torchani.Ensemble(nn_ensemble)

        if species_converter is None:
            self.model = torchani.nn.Sequential(aev_computer, nn).to(device)
        else:
            self.model = torchani.nn.Sequential(species_converter, aev_computer, nn).to(device)

        for il, layer in enumerate(self.model):
            if re.search(r"^Ensemble", str(layer)):
                log.debug("Network layer found number {}".format(il))
                self._network_layer_indx = il

    def complete_model(self,
                       model: Union[torchani.nn.Sequential, None] = None,
                       energy_shifter: Union[torchani.utils.EnergyShifter, None] = None,
                       filename: str = "complete_ani_model.pt" ):
        """
        Function to add energy shifter to trianed model. The energy shifter adds the atomic self energies to the
        model predictions of the interaction energies.
        :param model: torchani.nn.Sequential - trained model
        :param energy_shifter: torchani.utils.EnergyShifter - energy shifter to add atomic self energies back in
        :param filename: str - filename to save the complete model in
        :return: None
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is None:
            model = self.model

        if energy_shifter is None:
            energy_shifter = self.energy_shifter

        #TODO: we should think about adding species convert to the complete network for easier use of the calculator
        # https://aiqm.github.io/torchani/api.html#torchani.SpeciesConverter

        if self.isparallel is False:
            self.model = torchani.nn.Sequential(*[m for m in model], energy_shifter).to(device)
        else:
            self.model = torchani.nn.Sequential(*[m for m in model.module], energy_shifter).to(device)
            self.run_in_data_parallel()

        self.completed_model = True

        torch.save(
            {
                "nn": self.model.state_dict(),
            },
            filename,
        )

    def load_pretrained_on_to_model(self, pretrained: str, model_key: Union[None, str] = None):
        """
        Function to load a pretrained model on to the equivalent initialized architecture
        :param pretrained: str - file and path to load weights from
        :return: None
        """
        log = logging.getLogger(__name__)

        self.pretrained = pretrained

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is None:
            log.error("ERROR - must initialize model first with equivalent architecture")

        if self.isbayesian is True:
            log.info("Current model is Bayesian")
        else:
            log.info("Current model is not Bayesian")

        if self.is_multimodel_ensemble is True:
            log.info("Current model is a multi model ensemble")
        else:
            log.info("Current model is a single model")

        log.info(f"Loading pre-trained weights from {self.pretrained} on to current model on "
                 f"layer{self._network_layer_indx}")

        # Load weights and bias on the default network. ANI network is the first element in the object zero based.
        self.model = self.model.to(device)

        # if pretrained contains only the weight and biases etc .....

        if model_key is None:

            try:
                self.model[self._network_layer_indx].load_state_dict(torch.load(pretrained,
                                                                                map_location=torch.device(device)
                                                                                )
                                                                     )
                self.model[self._network_layer_indx].eval()

            except RuntimeError as rerr:
                # HACK: this allows us to load our old models with 1 BNN outside the ensemble wrapper
                try:
                    self.model[self._network_layer_indx][0].load_state_dict(torch.load(pretrained,
                                                                                    map_location=torch.device(device)
                                                                                    )
                                                                         )
                    self.model[self._network_layer_indx].eval()

                except RuntimeError as rerr2:
                    log.error(f"ERROR - loading pretrained model fails. Likely the ensemble is of different sizes please "
                              f"check.\nORIGINAL ERROR:\n{rerr}\n-----\n")
                    raise rerr2

        # if pretrained contains other things as well .....
        else:
            checkpoint = torch.load(pretrained, map_location=torch.device(device))

            try:
                self.model[self._network_layer_indx].load_state_dict(checkpoint[model_key])
                self.model[self._network_layer_indx].eval()

            except RuntimeError as rerr:
                # HACK: this allows us to load our old models with 1 BNN outside the ensemble wrapper
                try:
                    self.model[self._network_layer_indx][0].load_state_dict(checkpoint[model_key])
                    self.model[self._network_layer_indx].eval()

                except RuntimeError as rerr2:
                    log.error(
                        f"ERROR - loading pretrained model fails. Likely the ensemble is of different sizes please "
                        f"check.\nORIGINAL ERROR:\n{rerr}\n-----\n")
                    raise rerr2
            except KeyError as kerr:
                log.error(f"\nKEY ERROR - please provide a valid key on --pretrained_model_key option for the "
                          f"pretrained model file {pretrained}")
                raise kerr

    def set_schedulers(self, SGD, AdamW, sgd=None, adamw=None):
        """
        Function to set schedulers for training. Note you must pass in the optimizers for this to run correctly.
        :param SGD: torch.optim.SGD - optimizer instance for SGD
        :param AdamW: torch.optim.AdamW - instance of the AdamW optimizer
        :param sgd: torch.optim.lr_scheduler - instance of an optimizer scheduler of learning rate for sgd
        :param adamw: torch.optim.lr_scheduler - instance of optimizer scheduler of learning rate for admaw
        :return: None
        """

        if sgd is None:
            self.sgd_ani_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                SGD, factor=0.1, patience=100, threshold=0
            )
        else:
            self.sgd_ani_scheduler = sgd

        if adamw is None:
            self.adamw_ani_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                AdamW, factor=0.1, patience=100, threshold=0
            )
        else:
            self.adamw_ani_scheduler = adamw

    def transfer_dnn_to_bnn(self, bnn_prior: dict,
                            params_prior: dict,
                            bayesian_depth: int = None,
                            set_rho_explicitly: bool = True,
                            set_prior_explicitly: bool = True):
        """
        Function to convert ani to BNN to a set depth on the network
        :param bnn_prior: dict - output of chosen prior from bnn_priors
        :param params_prior : dict - dictionary keys are layers named by numbers as in model.named_parameters()
                              and to sub-dictionary where keys of weight_mean, weight_std, bias_mean and bias_std set
                              prior
        :param bayesian_depth : int - depth of layers to convert to bayesian
        :param set_rho_explicitly: bool - if params prior is pass use it to set the rho array explicitly
        :param set_prior_explicitly: bool - if params prior is pass use it to set the prior arrays explicitly
        return None
        """

        log = logging.getLogger(__name__)

        self.bnn_prior = bnn_prior
        self.params_prior = params_prior

        if self.isbayesian is False:
            log.info("Converting model DNN to BNN")

            #TODO: For some reason this tries to loop twice for BNNs we should have a better way to handle this.
            try:
                convert_dnn_to_bnn(model=self.model,
                                   bnn_prior=bnn_prior,
                                   params_prior=params_prior,
                                   bayesian_depth=bayesian_depth,
                                   set_rho_explicitly=set_rho_explicitly,
                                   set_prior_explicitly=set_prior_explicitly,
                                   prior_key=""
                                   )
                log.info("Model DNN converted to BNN")
                self.bayesian = True
            except AttributeError as aerr:
                log.warning(f"\n-----\nWARNING - Failed DNN to BNN conversion\n{aerr}\n-----\n")
                raise aerr

        else:
            log.info("Model is already Bayesian")