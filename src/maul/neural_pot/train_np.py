#!/usr/bi/env python

import argparse
import logging
import torch
import torchani
import re
import os
import json
import numpy as np
import pandas as pd
import random
from maul.neural_pot.maul_library.network_classes import (
    TorchNetwork,
    ANI,
    bnn_priors,
    convert_dnn_to_bnn,
    is_bayesian_network,
    update_bnn_variance,
)
from maul.neural_pot.maul_library import network_data_loader
from maul.neural_pot.maul_library import network_train
from maul.neural_pot.maul_library import network_test
from maul.neural_pot.maul_library import network_utilities
from maul.neural_pot.maul_library.network_utilities import kcalmol2hartree
from datetime import datetime
from torchani.units import hartree2kcalmol

__version__ = "1.0"
__authors__ = "James McDonagh, Clyde Fare, Ravikanth Tadikonda and Zeynep Sumer"
__contact__ = "james.mcdonagh@uk.ibm.com, clyde.fare@ibm.com, ravikanth.tadikonda@stfc.ac.uk and zsumer@ibm.com"
__organisation__ = "IBM Research Europe, Hartree Centre, Daresbury Laboratory"
__created__ = "Aug, 2024"
__description__ = """
                ANI1-x type potential generator and DNN ---> BNN converter
                """
__title__ = os.path.basename(__file__)


def setup_logger(cwd: str = os.getcwd(), loglev: str = "INFO"):
    """
    Make logger setup with file and screen handler
    :param cwd: str - the working directory you want the log file to be saved in
    :param loglev: str - logging level
    :return: None
    """
    # set log level from user
    intloglev = getattr(logging, loglev)
    try:
        intloglev + 1
    except TypeError:
        print(
            "ERROR - cannot convert loglev to numeric value using default of 20 = INFO"
        )
        with open("error_logging.log", "w+") as logerr:
            logerr.write(
                "ERROR - cannot convert loglev to numeric value using default of 20 = INFO"
            )
        intloglev = 20

    # Format routine the message comes from, the leve of the information debug, info, warning, error, critical
    # writes all levels to teh log file Optimizer.log
    logging.raiseExceptions = True
    log = logging.getLogger()
    log.setLevel(intloglev)
    pathlog = os.path.join(cwd, "{}.log".format(__title__.split(".")[0]))

    # File logging handle set up
    filelog = logging.FileHandler("{}".format(pathlog), mode="w")
    filelog.setLevel(intloglev)
    fileformat = logging.Formatter(
        "%(levelname)s - %(name)s - %(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S"
    )
    filelog.setFormatter(fileformat)

    # Setup handle for screen printing only prints info and above not debugging info
    screen = logging.StreamHandler()
    screen.setLevel(10)
    screenformat = logging.Formatter("%(message)s")
    screen.setFormatter(screenformat)

    # get log instance
    log.addHandler(screen)
    log.addHandler(filelog)

    log.info(
        "The handlers {} logging level {} {}".format(log.handlers, loglev, intloglev)
    )
    log.info("Started {}\n".format(datetime.now()))


def train_neural_potential(
    radial_cutoff: float = 5.2000e00,
    theta_max_rad: float = 3.33794218e00,
    angular_cutoff: float = 3.5000e00,
    etar: float = 1.6000000e01,
    etaa: float = 8.000000,
    zeta: float = 3.2000000e01,
    radial_steps: int = 16,
    angular_radial_steps: int = 4,
    theta_steps: int = 8,
    species_order: list = ["H", "C", "N", "O"],
    self_energies: list = [-0.6010, -38.0832, -54.7078, -75.1945],
    data: str = "../../data/ANI-1_release",
    no_reload: bool = False,
    checkpoint_filename: str = "ani_model_checkpoint.pt",
    data_pkl_path: str = "ensemble_0.pkl",
    learning_curve_csv_filename: str = "learning_curve.csv",
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    random_seed: float = 15234,
    forces: bool = False,
    force_scalar: float = 1.0,
    max_epochs: int = 100000,
    species_indicies: list = ["H", "C", "N", "O"],
    early_stopping_learning_rate: float = 1.0e-9,
    params_prior: str = None,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    posterior_mu_init: float = 0.0,
    posterior_rho_init: float = -3.0,
    reparameterization: bool = False,
    flipout: bool = False,
    moped: bool = False,
    moped_delta: float = 0.2,
    moped_enable: bool = False,
    moped_init_model: str = "dnn_v1_moped_init.pt",
    bayesian_depth: int = None,
    lml: bool = False,
    use_schedulers: bool = False,
    train_mc_runs: int = 2,
    test_mc_runs: int = 2,
    kl_weight: float = 0.0,
    min_rmse: float = 0.0,
    load_pretrained_parameters_dnn: str = None,
    load_pretrained_parameters_bnn: str = None,
    pretrained_model_key: str = "nn",
    update_bnn_variance: float = None,
    name: str = "dnn_name",
    mutate_index: int = None,
    no_mu_opt: bool = False,
    no_sig_opt: bool = False,
    train_ani_dnn: bool = False,
    train_ani_bnn: bool = False,
    ensemble: int = 1,
    set_rho_explicit: bool = False,
    set_prior_explicit: bool = False,
    use_cuaev: bool = False,
    loglevel: str = "INFO",
    parallel: bool = False,
    reset_lr: bool = False,
) -> None:
    """
    Main run function to train DNN ANI or BNN ANI. There are a lot of options here that change how the model is trained and the model type.

    :param radial_cutoff: float - Radial cutoff distance
    :param theta_max_rad: float - Theta distance cutoff
    :param angular_cutoff: float - Angular cutoff in radians
    :param etar: float - Eta in the radial function determines the guassians width
    :param etaa: float - Eta in the angular function determines the guassians width for the radial portion
    :param zeta: float - Zeta in the angular function determines the guassians width for the angular portion
    :param radial_steps: int - Number of radial shifts moves the peak of the gaussian
    :param angular_radial_steps: int - Number of angular shifts moves the peak of the gaussian
    :param theta_steps: int - Angle steps
    :param species_order: list - Species order at atomic symbol
    :param self_energies: list - Self energies in the order of species order
    :param data: str - Data directory where HDF5 file of training and testing data can be found
    :param no_reload: bool - Tell the code to reload data don't use previous loading data
    :param checkpoint_filename: str - initial weights path
    :param data_pkl_path: str - if pickle file already exists, write the name
    :param learning_curve_csv_filename: str - best weights path
    :param batch_size: int - Batch size for training
    :param learning_rate: float - Learning rate
    :param train_size: float - Training dataset size
    :param validation_size: float - Validation dataset size
    :param random_seed: float - random seed tp initialize
    :param forces: bool - Train set has forces and use them
    :param force_scalar: float - the weight to apply to the forces part of the loss when training
    :param max_epochs: int - maximum number of epochs
    :param species_indicies: list - Whether to read species by atomic number (ani 1x data) or label mapped to network index (ani 1 data)
    :param early_stopping_learning_rate: float - Early stopping if this learning rate is met
    :param params_prior: str - 'ani' or Json file with prior parameters stored under appropriate keys for setting each individually
    :param prior_mu: float - Prior mean
    :param prior_sigma: float - Prior standard deviation
    :param posterior_mu_init: float - Posterior mean initialization
    :param posterior_rho_init: float - Posterior denisty initialization
    :param reparameterization: bool - Use the reparameterization version of the bayesian layers
    :param flipout: bool - Use the flipout version of the bayesian layers
    :param moped: bool - Use the moped priors
    :param moped_delta: float - MOPED delta value
    :param moped_enable: bool - Initialize mu/sigma from the dnn weights
    :param moped_init_model: str - DNN model to intialize MOPED method filename and path /path/to/dnn_v1_moped_init.pt
    :param bayesian_depth: int - Use bayesian layers to a depth of n (None means all)
    :param lml: bool - Whether to replace the MSE loss with LML loss for the BNN
    :param use_schedulers: bool - Use lr schedulers for adam and sgd
    :param train_mc_runs: int - Number of Monte Carlo runs during training of the Bayesian potential
    :param test_mc_runs: int - Number of Monte Carlo runs during testing of the Bayesian potential
    :param kl_weight: float - KL factor
    :param min_rmse: float - stop the training if the rmse is reached
    :param load_pretrained_parameters_dnn: str - Path and file name to load pre-trained DNN parameters from
    :param load_pretrained_parameters_bnn: str - Path and file name to load pre-trained BNN parameters from
    :param pretrained_model_key: str - if there is a key the model weights and biases are stored give it here
    :param update_bnn_variance: float - If you load a pretrained BNN but want to update the varience of the parameters set a fraction between 0.0 and 1.0.
    :param name: str - Name for the model sor reference later on
    :param mutate_index: int - Permute dataset swapping examples in training and validation set using this index as a random seed. None will not permute the data set beyond the normal loading
    :param no_mu_opt: bool - Whether to omit training the mean weights of the BNN
    :param no_sig_opt: bool - Whether to omit training the sigma weights of the BNN
    :param train_ani_dnn: bool - Train a DNN
    :param train_ani_bnn: bool - Train a BNN
    :param ensemble: int - Number of concurrent models to train as an ensemble prediction
    :param set_rho_explicit: bool - If params prior is being used will set rho values using the params prior
    :param set_prior_explicit: bool - If params prior is being used will set prior values using the params prior
    :param use_cuaev: bool - Use the cuda ave codes note these are installed separately from ani library
    :param loglevel: str - log level
    :param parallel: bool - Run using data parallel pytorch stratergy
    :param reset_lr: bool - do not use the lr from checkpoint
    """

    setup_logger(os.getcwd(), loglevel)
    log = logging.getLogger(__name__)

    log.info(
        "\n--------------------------------------------------------------------------------------------------\n"
        "\nAuthors       : {}\nOrganisation  : {}\nCreated On    : {}\n"
        "Program       : {}\nVersion       : {}\nDescription   : {}\n"
        "---------------------------------------------------------------------------------------------------\n".format(
            __authors__,
            __organisation__,
            __created__,
            __title__,
            __version__,
            __description__,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("\nDevices we will run on {}".format(device))
    log.info("Using Torhani version: {}".format(torchani.__version__))
    log.info("Torchani file: {}".format(torchani.__file__))
    log.info("Using random seed: {}\n".format(random_seed))

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if (
        len(species_indicies) == 1
        and species_indicies[0].strip().lower() == "periodic_table"
    ):
        species_indicies = "periodic_table"
    log.info("Species indicies: {}".format(species_indicies))

    if train_ani_bnn is True or train_ani_dnn is True:

        log.info(f"\nUse forces for training: {forces}\n")

    if params_prior is not None:
        if params_prior == "ani":
            if self_energies is None:
                model = torchani.models.ANI1x(periodic_table_index=True).to(device)
                self_energies = model[0].energy_shifter.self_energies

    if self_energies is not None:
        if not isinstance(self_energies, np.ndarray):
            self_energies = np.array(self_energies)

    ani_class = ANI(
        forces=forces, force_scalar=force_scalar, self_energies=self_energies
    )

    train_data, val_data, test_data = network_data_loader.load_ani_data(
        ani_class.energy_shifter,
        species_order,
        data,
        batch_size,
        train_size,
        validation_size,
        forces=forces,
        no_reload=no_reload,
        species_indicies=species_indicies,
        data_pkl_path=data_pkl_path,
        mutate_datasets=mutate_index,
        random_seed=random_seed,
    )

    ani_class.build_ani_dnn_model(
        radial_cutoff,
        theta_max_rad,
        angular_cutoff,
        etar,
        etaa,
        zeta,
        radial_steps,
        angular_radial_steps,
        theta_steps,
        species_order,
        networks=None,
        ensemble=ensemble,
        use_cuaev=use_cuaev,
        no_species_converter=False if species_indicies == "periodic_table" else True,
    )

    if load_pretrained_parameters_dnn is not None:
        ani_class.load_pretrained_on_to_model(
            load_pretrained_parameters_dnn, model_key=pretrained_model_key
        )

    log.info(f"\nIs initial network bayesian: {ani_class.isbayesian}")
    log.info(f"Train using forces for ANI {ani_class.uses_forces}\n")

    if train_ani_dnn is True:
        log.info("\nRequest to train DNN")
    elif train_ani_bnn is True:
        log.info("\nRequest to train BNN")
        bayesian_priors = bnn_priors(
            reparameterization,
            flipout,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            moped_enable,
            moped_delta,
        )
        params_prior = None

        if params_prior is not None and params_prior.strip().lower() != "ani":
            log.info(f"Params prior: {params_prior} loading")
            with open(params_prior, "r") as fin:
                params_prior = json.load(fin)

        elif params_prior is not None:
            if params_prior.strip().lower() == "ani":
                # In this case we are attempting to match the ANI 1x layers to our layers in terms of parameters
                # to do this we need to map the layers. ANI 1x uses certain keys in a fixed order we map those to
                # our network which uses integer keys. To do this we prepend the layer number of the Ensemble
                # and the number of the ensemble elements using a BNN we only have one member of the ensemble so it
                # is always 0.
                log.info(f"Params prior: is ani loading mean values of ani models")
                for il, layer in enumerate(ani_class.model):
                    if re.search(r"^Ensemble", str(layer)):
                        log.info("Network layer found number {}".format(il))
                        network_layer_number = il
                    log.debug(f"Layer: {layer}")
                params_prior = network_utilities.get_ani_parameter_distrbutions(
                    species_order=tuple(species_order),
                    prepend_key="{}.0.".format(network_layer_number),
                )
        else:
            log.info(
                f"Params prior not given or asked for 'ani' will initialize using the dnn."
            )
            params_prior = None

        # Note the options are to not set the parameters explicitly as the default is to set them explicitly
        # the function arguments though are set assuming the opposite hence not.
        # log.debug(f"{no_set_rho_explicit}, {not no_set_rho_explicit}\n"
        #         f"{no_set_rho_explicit}, {not no_set_rho_explicit}")
        ani_class.transfer_dnn_to_bnn(
            bnn_prior=bayesian_priors,
            params_prior=params_prior,
            set_rho_explicitly=set_rho_explicit,
            set_prior_explicitly=set_rho_explicit,
        )

        log.info(ani_class._network_layer_indx)
        if load_pretrained_parameters_bnn is not None:
            ani_class.load_pretrained_on_to_model(
                load_pretrained_parameters_bnn, model_key=pretrained_model_key
            )

        if update_bnn_variance is not None:
            log.info(
                f"\nUpdating BNN variance to {update_bnn_variance} fraction of the mean weight and bias"
            )

            updated_bnn_prior = bnn_priors(
                reparameterization,
                flipout,
                prior_mu,
                prior_sigma,
                posterior_mu_init,
                posterior_rho_init,
                moped_enable,
                update_bnn_variance,
            )

            update_bnn_variance(
                ani_class.model,
                bnn_prior=updated_bnn_prior,
                params_prior=None,
                bayesian_depth=bayesian_depth,
            )

            log.info(
                "BNN variance update completed. (NOTE: if you want to update with tests use the method in "
                "network_utilities\n"
            )

        log.info(f"bnn network is bayesian: {ani_class.isbayesian}\n")

    if parallel is True:
        ani_class.run_in_data_parallel()

    log.info(f"Is the model using parallelization: {ani_class.isparallel}")

    log.info(ani_class.model)

    ani_class.name = name

    log.info(f"Is network to be trained bayesian: {ani_class.isbayesian}")

    log.info("\n\nUntrained testing .....\n")
    ani_class.model.to(device)
    network_test.test(
        ani_class,
        test_data,
        mc_runs=test_mc_runs,
        plot=True,
        plot_name_unique_prepend="initial_",
    )
    log.info("Finalized untrained testing .....\n")

    log.info("\n\nBeginning training .....\n")
    network_train.train(
        ani_class,
        train_data,
        val_data,
        test_data,
        checkpoint_filename=checkpoint_filename,
        learning_curve_csv_filename=learning_curve_csv_filename,
        max_epochs=max_epochs,
        early_stopping_learning_rate=early_stopping_learning_rate,
        train_mc_runs=train_mc_runs,
        test_mc_runs=test_mc_runs,
        kl_weight=kl_weight,
        learning_rate=learning_rate,
        bayesian_depth=bayesian_depth,
        mu_opt=not no_mu_opt,
        sig_opt=not no_sig_opt,
        lml=lml,
        use_schedulers=use_schedulers,
        checkpoint_dnn=load_pretrained_parameters_dnn,
        checkpoint_bnn=load_pretrained_parameters_bnn,
        reset_lr=reset_lr,
        min_rmse=min_rmse,
    )

    log.info("Finalized training .....\n")

    log.info(f"Is network for test bayesian: {ani_class.isbayesian}")

    log.info("\n\nBeginning testing .....\n")
    network_test.test(ani_class, test_data, mc_runs=test_mc_runs, plot=True)
    log.info("Finalized testing .....\n")

    log.info(
        "Completing ANI model by adding energy shifter to include atomic self energies"
    )
    ani_class.complete_model()
    log.info(f"Is the ANI model complete: {ani_class.iscomplete}")

    log.info("\n\nPredicting H2 molecule energy.")
    coordin = torch.tensor(
        [[[0, 0, 0], [0, 0, 0.7]]], requires_grad=False, device=device
    )
    # In periodic table, C = 8 and H = 1
    if species_indicies == "periodic_table":
        specs = torch.tensor([[1, 1]], device=device)
    else:
        specs = torch.tensor([[0, 0]], device=device)

    if train_ani_dnn is True:
        ener = ani_class.model((specs, coordin)).energies
        log.info(f"H2 molecule energy predicted to be {ener.item()} Ha")

    elif train_ani_bnn is True:
        eners = []

        for m in range(test_mc_runs):
            log.debug(f"Sample {m}")
            e_tensor = ani_class.model((specs, coordin)).energies
            eners.append(e_tensor.item())

        ener = np.mean(eners)
        ener_u = np.std(eners)
        log.info(f"H2 molecule energy predicted to be {ener} uncertainty {ener_u} Ha")

    ani_class.save()
