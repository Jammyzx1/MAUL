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


def get_parse_opt():
    """
    Function to parsing command-line options
    :return : argparse.parser - Command line parameter parser
    """

    try:
        usage = "python bnn_np_main.py Required Parameters options .....\n"

        parser = argparse.ArgumentParser(
            description="Command line binary points script",
            usage=usage,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        group = parser.add_argument_group("Featurization options")

        group.add_argument(
            "--radial_cutoff",
            action="store",
            type=float,
            help="Radial cutoff distance",
            default=5.2000e00,
        )

        group.add_argument(
            "--theta_max_rad",
            action="store",
            type=float,
            help="Theta distance cutoff",
            default=3.33794218e00,
        )

        group.add_argument(
            "--angular_cutoff",
            action="store",
            type=float,
            help="Angular cutoff in radians",
            default=3.5000e00,
        )

        group.add_argument(
            "--etar",
            action="store",
            type=float,
            help="Eta in the radial function determines the guassians width",
            default=1.6000000e01,
        )

        group.add_argument(
            "--etaa",
            action="store",
            type=float,
            help="Eta in the angular function determines the guassians width for the radial portion",
            default=8.000000,
        )

        group.add_argument(
            "--zeta",
            action="store",
            type=float,
            help="Zeta in the angular function determines the guassians width for the angular portion",
            default=3.2000000e01,
        )

        group.add_argument(
            "--radial_steps",
            action="store",
            type=int,
            help="Number of radial shifts moves the peak of the gaussian",
            default=16,
        )

        group.add_argument(
            "--angular_radial_steps",
            action="store",
            type=int,
            help="Number of angular shifts moves the peak of the gaussian",
            default=4,
        )

        group.add_argument(
            "--theta_steps",
            action="store",
            type=int,
            help="Angle steps",
            default=8,
        )

        group.add_argument(
            "--species_order",
            action="store",
            nargs="*",
            type=str,
            help="Species order at atomic symbol",
            default=["H", "C", "N", "O"],
        )

        group.add_argument(
            "--self_energies",
            action="store",
            nargs="*",
            type=float,
            help="Self energies in the order of species order",
            default=[-0.6010, -38.0832, -54.7078, -75.1945],
        )

        group = parser.add_argument_group("Data options")

        group.add_argument(
            "--data",
            metavar="str",
            action="store",
            help="Data directory where HDF5 file of training and testing data can "
            "be found",
            default="../../data/ANI-1_release",
        )

        group.add_argument(
            "--no_reload",
            action="store_true",
            help="Tell the code to reload data don't use previous loading data",
            default=False,
        )

        group.add_argument(
            "--checkpoint_filename",
            type=str,
            default="ani_model_checkpoint.pt",
            help="initial weights path",
        )

        group.add_argument(
            "--data_pkl_path",
            metavar="str",
            action="store",
            help="if pickle file already exists, write the name",
            default="ensemble_0.pkl",
        )

        group.add_argument(
            "--learning_curve_csv_filename",
            type=str,
            default="learning_curve.csv",
            help="best weights path",
        )

        group.add_argument(
            "--batch_size",
            action="store",
            type=int,
            help="Batch size for training",
            default=1024,
        )

        group.add_argument(
            "--learning_rate",
            action="store",
            type=float,
            help="Learning rate",
            default=1e-4,
        )

        group.add_argument(
            "--train_size",
            action="store",
            type=float,
            help="Training dataset size",
            default=0.8,
        )

        group.add_argument(
            "--validation_size",
            action="store",
            type=float,
            help="Validation dataset size",
            default=0.1,
        )

        group.add_argument(
            "--random_seed",
            action="store",
            type=float,
            help="random seed tp initialize",
            default=15234,
        )

        group.add_argument(
            "--forces",
            action="store_true",
            help="Train set has forces and use them",
            default=False,
        )

        group.add_argument(
            "--force_scalar",
            action="store",
            help="the weight to apply to the forces part of the loss when training",
            default=1.0,
        )

        group.add_argument(
            "--max_epochs",
            action="store",
            help="maximum number of epochs",
            type=int,
            default=100000,
        )

        group.add_argument(
            "--species_indicies",
            action="store",
            help="Whether to read species by atomic number (ani 1x data) or label mapped to network index (ani 1 data)",
            nargs="+",
            type=str,
            default=["H", "C", "N", "O"],  # ["periodic_table"],
        )

        group.add_argument(
            "--early_stopping_learning_rate",
            action="store",
            default=1.0e-9,
            type=float,
            help="Early stopping if this learning rate is met",
        )

        group = parser.add_argument_group("BNN prior options")

        group.add_argument(
            "--params_prior",
            action="store",
            help="'ani' or Json file with prior parameters stored under appropriate keys for setting each individually",
            type=str,
            default=None,
        )

        group.add_argument(
            "--prior_mu", action="store", help="Prior mean", type=float, default=0.0
        )

        group.add_argument(
            "--prior_sigma",
            action="store",
            help="Prior standard deviation",
            type=float,
            default=1.0,
        )

        group.add_argument(
            "--posterior_mu_init",
            action="store",
            help="Posterior mean initialization",
            type=float,
            default=0.0,
        )

        group.add_argument(
            "--posterior_rho_init",
            action="store",
            help="Posterior denisty initialization",
            type=float,
            default=-3.0,
        )

        group.add_argument(
            "--reparameterization",
            action="store_true",
            default=False,
            help="Use the reparameterization version of the bayesian layers",
        )

        group.add_argument(
            "--flipout",
            action="store_true",
            default=False,
            help="Use the flipout version of the bayesian layers",
        )

        group.add_argument(
            "--moped",
            action="store_true",
            default=False,
            help="Use the moped priors",
        )

        group.add_argument(
            "--moped_delta",
            action="store",
            help="MOPED delta value",
            type=float,
            default=0.2,
        )

        group.add_argument(
            "--moped_enable",
            action="store_true",
            default=False,
            help="Initialize mu/sigma from the dnn weights",
        )

        group.add_argument(
            "--moped_init_model",
            dest="moped_init_model",
            type=str,
            default="dnn_v1_moped_init.pt",
            help="DNN model to intialize MOPED method filename and path /path/to/dnn_v1_moped_init.pt",
        )

        group.add_argument(
            "--bayesian_depth",
            action="store",
            help="Use bayesian layers to a depth of n (None means all)",
            type=int,
            default=None,
        )

        group.add_argument(
            "--lml",
            action="store_true",
            help="Whether to replace the MSE loss with LML loss for the BNN",
            default=False,
        )

        group = parser.add_argument_group("Training options")

        group.add_argument(
            "--use_schedulers",
            action="store_true",
            default=False,
            help="Use lr schedulers for adam and sgd",
        )

        group.add_argument(
            "--train_mc_runs",
            action="store",
            help="Number of Monte Carlo runs during training of the Bayesian "
            "potential",
            type=int,
            default=2,
        )

        group.add_argument(
            "--test_mc_runs",
            action="store",
            help="Number of Monte Carlo runs during testing of the Bayesian "
            "potential",
            type=int,
            default=2,
        )

        group.add_argument(
            "--kl_weight", action="store", help="KL factor", type=float, default=0.0
        )

        group.add_argument(
            "--min_rmse",
            action="store",
            help="stop the training if the rmse is reached",
            type=float,
            default=0.0,
        )

        group.add_argument(
            "--load_pretrained_parameters_dnn",
            action="store",
            help="Path and file name to load pre-trained DNN parameters from",
            type=str,
            default=None,
        )

        group.add_argument(
            "--load_pretrained_parameters_bnn",
            action="store",
            help="Path and file name to load pre-trained BNN parameters from",
            type=str,
            default=None,
        )

        group.add_argument(
            "--pretrained_model_key",
            action="store",
            help="if there is a key the model weights and biases are stored give it here",
            type=str,
            default="nn",
        )

        group.add_argument(
            "--update_bnn_variance",
            action="store",
            help="If you load a pretrained BNN but want to update the varience of the parameters set a fraction between "
            "0.0 and 1.0.",
            type=float,
            default=None,
        )

        group.add_argument(
            "--name",
            action="store",
            help="Name for the model sor reference later on",
            type=str,
            default="dnn_name",
        )

        group.add_argument(
            "--mutate_index",
            action="store",
            help="Permute dataset swapping examples in training and validation set using this index as a random"
            "seed. None will not permute the data set beyond the normal loading",
            type=int,
            default=None,
        )
        group.add_argument(
            "--no_mu_opt",
            action="store_true",
            help="Whether to omit training the mean weights of the BNN",
            default=False,
        )

        group.add_argument(
            "--no_sig_opt",
            action="store_true",
            help="Whether to omit training the sigma weights of the BNN",
            default=False,
        )

        group = parser.add_argument_group("Running options")

        group.add_argument(
            "--train_ani_dnn", action="store_true", help="Train a DNN", default=False
        )

        group.add_argument(
            "--train_ani_bnn", action="store_true", help="Train a BNN", default=False
        )

        group.add_argument(
            "--ensemble",
            action="store",
            type=int,
            help="Number of concurrent models to train as an ensemble prediction",
            default=1,
        )

        group.add_argument(
            "--set_rho_explicit",
            action="store_true",
            help="If params prior is being used will set rho values using the params prior",
            default=False,
        )

        group.add_argument(
            "--set_prior_explicit",
            action="store_true",
            help="If params prior is being used will set prior values using the params prior",
            default=False,
        )

        group.add_argument(
            "--use_cuaev",
            action="store_true",
            help="Use the cuda ave codes note these are installed separately from ani library",
            default=False,
        )

        group = parser.add_argument_group("Logging arguments")

        group.add_argument(
            "--loglevel", action="store", default="INFO", help="log level"
        )

        group.add_argument(
            "--parallel",
            action="store_true",
            help="Run using data parallel pytorch stratergy",
            default=False,
        )

        group.add_argument(
            "--reset_lr",
            action="store_true",
            default=False,
            help="do not use the lr from checkpoint",
        )

        opt = parser.parse_args()

        return opt

    except argparse.ArgumentError as err:

        print(
            "\nERROR - command line arguments are ill defined please check the arguments\n"
        )
        raise err


def run():
    """
    Main run function to train DNN ANI or BNN ANI
    :return: None
    """

    opt = get_parse_opt()

    setup_logger(os.getcwd(), opt.loglevel)
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

    log.info("Command line input =\n\t{}".format(opt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("\nDevices we will run on {}".format(device))
    log.info("Using Torhani version: {}".format(torchani.__version__))
    log.info("Torchani file: {}".format(torchani.__file__))
    log.info("Using random seed: {}\n".format(opt.random_seed))

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)

    if (
        len(opt.species_indicies) == 1
        and opt.species_indicies[0].strip().lower() == "periodic_table"
    ):
        opt.species_indicies = "periodic_table"
    log.info("Species indicies: {}".format(opt.species_indicies))

    if opt.train_ani_bnn is True or opt.train_ani_dnn is True:

        log.info(f"\nUse forces for training: {opt.forces}\n")

    if opt.params_prior is not None:
        if opt.params_prior == "ani":
            if opt.self_energies is None:
                model = torchani.models.ANI1x(periodic_table_index=True).to(device)
                opt.self_energies = model[0].energy_shifter.self_energies

    if opt.self_energies is not None:
        if not isinstance(opt.self_energies, np.ndarray):
            opt.self_energies = np.array(opt.self_energies)

    ani_class = ANI(
        forces=opt.forces,
        force_scalar=opt.force_scalar,
        self_energies=opt.self_energies,
    )

    train_data, val_data, test_data = network_data_loader.load_ani_data(
        ani_class.energy_shifter,
        opt.species_order,
        opt.data,
        opt.batch_size,
        opt.train_size,
        opt.validation_size,
        forces=opt.forces,
        no_reload=opt.no_reload,
        species_indicies=opt.species_indicies,
        data_pkl_path=opt.data_pkl_path,
        mutate_datasets=opt.mutate_index,
        random_seed=opt.random_seed,
    )

    ani_class.build_ani_dnn_model(
        opt.radial_cutoff,
        opt.theta_max_rad,
        opt.angular_cutoff,
        opt.etar,
        opt.etaa,
        opt.zeta,
        opt.radial_steps,
        opt.angular_radial_steps,
        opt.theta_steps,
        opt.species_order,
        networks=None,
        ensemble=opt.ensemble,
        use_cuaev=opt.use_cuaev,
        no_species_converter=False
        if opt.species_indicies == "periodic_table"
        else True,
    )

    if opt.load_pretrained_parameters_dnn is not None:
        ani_class.load_pretrained_on_to_model(
            opt.load_pretrained_parameters_dnn, model_key=opt.pretrained_model_key
        )

    log.info(f"\nIs initial network bayesian: {ani_class.isbayesian}")
    log.info(f"Train using forces for ANI {ani_class.uses_forces}\n")

    if opt.train_ani_dnn is True:
        log.info("\nRequest to train DNN")
    elif opt.train_ani_bnn is True:
        log.info("\nRequest to train BNN")
        bayesian_priors = bnn_priors(
            opt.reparameterization,
            opt.flipout,
            opt.prior_mu,
            opt.prior_sigma,
            opt.posterior_mu_init,
            opt.posterior_rho_init,
            opt.moped_enable,
            opt.moped_delta,
        )
        params_prior = None

        if opt.params_prior is not None and opt.params_prior.strip().lower() != "ani":
            log.info(f"Params prior: {opt.params_prior} loading")
            with open(opt.params_prior, "r") as fin:
                params_prior = json.load(fin)

        elif opt.params_prior is not None:
            if opt.params_prior.strip().lower() == "ani":
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
                    species_order=tuple(opt.species_order),
                    prepend_key="{}.0.".format(network_layer_number),
                )
        else:
            log.info(
                f"Params prior not given or asked for 'ani' will initialize using the dnn."
            )
            params_prior = None

        # Note the options are to not set the parameters explicitly as the default is to set them explicitly
        # the function arguments though are set assuming the opposite hence not.
        # log.debug(f"{opt.no_set_rho_explicit}, {not opt.no_set_rho_explicit}\n"
        #         f"{opt.no_set_rho_explicit}, {not opt.no_set_rho_explicit}")
        ani_class.transfer_dnn_to_bnn(
            bnn_prior=bayesian_priors,
            params_prior=params_prior,
            set_rho_explicitly=opt.set_rho_explicit,
            set_prior_explicitly=opt.set_rho_explicit,
        )

        log.info(ani_class._network_layer_indx)
        if opt.load_pretrained_parameters_bnn is not None:
            ani_class.load_pretrained_on_to_model(
                opt.load_pretrained_parameters_bnn, model_key=opt.pretrained_model_key
            )

        if opt.update_bnn_variance is not None:
            log.info(
                f"\nUpdating BNN variance to {opt.update_bnn_variance} fraction of the mean weight and bias"
            )

            updated_bnn_prior = bnn_priors(
                opt.reparameterization,
                opt.flipout,
                opt.prior_mu,
                opt.prior_sigma,
                opt.posterior_mu_init,
                opt.posterior_rho_init,
                opt.moped_enable,
                opt.update_bnn_variance,
            )

            update_bnn_variance(
                ani_class.model,
                bnn_prior=updated_bnn_prior,
                params_prior=None,
                bayesian_depth=opt.bayesian_depth,
            )

            log.info(
                "BNN variance update completed. (NOTE: if you want to update with tests use the method in "
                "network_utilities\n"
            )

        log.info(f"bnn network is bayesian: {ani_class.isbayesian}\n")

    if opt.parallel is True:
        ani_class.run_in_data_parallel()

    log.info(f"Is the model using parallelization: {ani_class.isparallel}")

    log.info(ani_class.model)

    ani_class.name = opt.name

    log.info(f"Is network to be trained bayesian: {ani_class.isbayesian}")

    log.info("\n\nUntrained testing .....\n")
    ani_class.model.to(device)
    network_test.test(
        ani_class,
        test_data,
        mc_runs=opt.test_mc_runs,
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
        checkpoint_filename=opt.checkpoint_filename,
        learning_curve_csv_filename=opt.learning_curve_csv_filename,
        max_epochs=opt.max_epochs,
        early_stopping_learning_rate=opt.early_stopping_learning_rate,
        train_mc_runs=opt.train_mc_runs,
        test_mc_runs=opt.test_mc_runs,
        kl_weight=opt.kl_weight,
        learning_rate=opt.learning_rate,
        bayesian_depth=opt.bayesian_depth,
        mu_opt=not opt.no_mu_opt,
        sig_opt=not opt.no_sig_opt,
        lml=opt.lml,
        use_schedulers=opt.use_schedulers,
        checkpoint_dnn=opt.load_pretrained_parameters_dnn,
        checkpoint_bnn=opt.load_pretrained_parameters_bnn,
        reset_lr=opt.reset_lr,
        min_rmse=opt.min_rmse,
    )

    log.info("Finalized training .....\n")

    log.info(f"Is network for test bayesian: {ani_class.isbayesian}")

    log.info("\n\nBeginning testing .....\n")
    network_test.test(ani_class, test_data, mc_runs=opt.test_mc_runs, plot=True)
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
    if opt.species_indicies == "periodic_table":
        specs = torch.tensor([[1, 1]], device=device)
    else:
        specs = torch.tensor([[0, 0]], device=device)

    if opt.train_ani_dnn is True:
        ener = ani_class.model((specs, coordin)).energies
        log.info(f"H2 molecule energy predicted to be {ener.item()} Ha")

    elif opt.train_ani_bnn is True:
        eners = []

        for m in range(opt.test_mc_runs):
            log.debug(f"Sample {m}")
            e_tensor = ani_class.model((specs, coordin)).energies
            eners.append(e_tensor.item())

        ener = np.mean(eners)
        ener_u = np.std(eners)
        log.info(f"H2 molecule energy predicted to be {ener} uncertainty {ener_u} Ha")

    ani_class.save()


if __name__ == "__main__":
    run()

# Example commandline
# BNN -
# --data Data_ani --max_epoch 100
# --reparameterization --mc_runs 4
# --load_pretrained_parameters_bnn maul1_weights/maul1_weights.pt
# --train_ani_bnn --batch_size 1024 --name test
