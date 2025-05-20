import glob
import os
from typing import Any, NoReturn, Union
import numpy as np
import pickle
import itertools
import torchani
import logging


__title__ = os.path.basename(__file__)


def data_merge(*iterators):
    """
    Function to load and merger ANI dataset
    :param iterators: iterator - lists of data load using ani data loader
    """

    empty = {}
    for values in itertools.zip_longest(*iterators, fillvalue=empty):
        for value in values:
            if value is not empty:
                yield value


def load_ani_data(
    energy_shifter: torchani.utils.EnergyShifter,
    species_order: list,
    data: str,
    batch_size: int,
    train_size: float,
    val_size: float,
    forces: bool = False,
    no_reload: bool = False,
    species_indicies: Union[str, list, tuple] = "periodic_table",
    data_pkl_path: str = "ani_datasets.pkl",
    mutate_datasets: Union[None, int] = None,
    random_seed: int = 15234,
) -> (
    torchani.data.TransformableIterable,
    torchani.data.TransformableIterable,
    torchani.data.TransformableIterable,
):

    """
    Function to load and merger ANI dataset
    :param energy_shifter: torchani.utils.EnergyShifter -
    :param species_order: itertable - list of element symbols ["H", "C", "N", "O"]
    :param data: str - path to find HDF5 files suitable for ani dataloader with extension *.h5
    :param batch_size: int - number of sample to train on each batch
    :param train_size: float - fraction of the data for training
    :param val_size: float - fraction of the data for validation
    :param forces: bool - use forces in the training for not
    :param no_reload: bool - use the previously stored data or reload
    :param species_indicies: str, iterable - either periodic table to convert species from atomic number to network
                                             number or iterable of species ["H", "C", "N", "O"] to specify atomic label
                                             mapping to the network number. The latter is need to load ani 1 data.
    :param data_pkl_path: str - path and name to save the pickle file in
    :param mutate_datasets: None, int - if None return the datasets split train, test validate as they are. If an int
                                        mutate training and validation swapping some entries based on the int as a
                                        random seed value. Useful for ensemble training where each ensemble member may
                                        need to be trained in a different sub-set (Note each ensemble member should be
                                        treated independetly with a unique index given as the argument here, the code
                                        WILL NOT provide different data to members of the ensemble class you need to
                                        construct the ensemble after training the models)
    :param random_seed: int - reset random seed for numpy if mutate is used
    :return: (torchani.data.TransformableIterable, torchani.data.TransformableIterable,
            torchani.data.TransformableIterable) - merged training, validation, and test data
    """

    log = logging.getLogger(__name__)

    train_iterators = []
    val_iterators = []
    test_iterators = []

    if no_reload is False:
        log.info("If {} is present will reload the file".format(data_pkl_path))
    else:
        log.info(
            "Will load data fresh and not use any previous stored data no_reload option given"
        )

    log.info("Batch size: {}".format(batch_size))

    if os.path.isfile(data_pkl_path) and no_reload is False:
        log.info(
            "Loading dataset to maintain consistent train, test, validation "
            "splitting"
        )

        with open(data_pkl_path, "rb") as fin:
            dataset = pickle.load(fin)

            try:
                train_data = dataset["training"].collate(batch_size).cache()
            except KeyError as kerr:
                log.error("ERROR - loading training data from {}".format(data_pkl_path))
                raise (kerr)

            try:
                test_data = dataset["test"].collate(batch_size).cache()
            except KeyError as kerr:
                log.error("ERROR - loading test data from {}".format(data_pkl_path))
                raise (kerr)

            try:
                val_data = dataset["validation"].collate(batch_size).cache()
            except KeyError as kerr:
                log.error(
                    "ERROR - loading validation data from {}".format(data_pkl_path)
                )
                raise (kerr)

            try:
                energy_shifter.self_energies = dataset["self_energies"]
            except KeyError as kerr:
                log.error(
                    "ERROR - loading self energy data from {}".format(data_pkl_path)
                )
                raise (kerr)

            if mutate_datasets is not None:
                train_data, val_data = mutate(
                    train_data,
                    val_data,
                    ensemble_index=mutate_datasets,
                    rseed=random_seed,
                )
    else:
        # load data sets
        store_se = False
        if energy_shifter.self_energies is not None:
            log.info("The energy shifter provided is already initialized")
            log.info(
                "\n-----\nWill use the values of this initialized energy shifter rather than "
                "setting new ones for this data set.\nIf you want new one to be set "
                "pass in `torchani.EnergyShifter(None)`.\nWe assume that the "
                "energies are in the same order as species order, please check the output dictionary\n"
                "------\n"
            )

            # NOTE: to keep using the self energies as we defined them we have to pass in a dictionary to
            # subtract self energies with the key of the neural network index. This means we use se for the data
            # preparation if the self energies are given.
            se = {
                k: energy_shifter.self_energies[ith]
                for ith, k in enumerate(species_order)
            }
            log.info(
                "\n----- PLEASE CHECK ------\nPlease check this dictionary is correct:\n{}\n------".format(
                    se
                )
            )
        else:
            log.info("No energy shifter given setting up a new energy shifter instance")
            # NOTE: if self energies are got given as input then they are found through linear regression we then need
            # to store these so they can be reused as needed. This requires some swapping of se and energy_shifter. i.e.
            # if we define as input he self energies then we need to store those with the model, if we don't then we
            # need to store the linear regression values from the energy shifter.
            se = energy_shifter
            store_se = True

        files = sorted(glob.glob(os.path.join(data, "*.h5")))
        if len(files) > 1:
            for ith, finh5 in enumerate(files):
                log.info("loading {}".format(finh5))

                if forces is False:
                    log.info("Will not use forces data (--forces not used)")
                    training, validation, test = (
                        torchani.data.load(finh5)
                        .subtract_self_energies(se, species_order)
                        .species_to_indices(species_indicies)
                        .shuffle()
                        .split(train_size, val_size, None)
                    )
                else:
                    log.info("Will use forces data (--forces used)")
                    training, validation, test = (
                        torchani.data.load(finh5, additional_properties=("forces",))
                        .subtract_self_energies(se, species_order)
                        .species_to_indices(species_indicies)
                        .shuffle()
                        .split(train_size, val_size, None)
                    )

                log.info("{} loaded".format(finh5))
                train_iterators.append(training)
                val_iterators.append(validation)
                test_iterators.append(test)

            if any(
                len(ent) == 0
                for ent in [train_iterators, val_iterators, test_iterators]
            ):
                log.warning(
                    "WARNING - there appear to be no *.h5 files given as input in "
                    "{} "
                    "please check inputs.".format(data)
                )
                raise RuntimeError("No inputs")

            # Lists are needed not iterators so that the data can be pickeled
            # May be this leads to memory issues in which case we will need
            # to be careful here
            train_merger = list(data_merge(*train_iterators))
            log.info("Training set size: {}".format(len(train_merger)))
            train_iterable = torchani.data.TransformableIterable(train_merger)
            log.info("Training data merged")

            val_merger = list(data_merge(*val_iterators))
            log.info("Validation set size: {}".format(len(val_merger)))
            val_iterable = torchani.data.TransformableIterable(val_merger)
            log.info("Validation data merged")

            test_merger = list(data_merge(*test_iterators))
            log.info("Test set size: {}".format(len(test_merger)))
            test_iterable = torchani.data.TransformableIterable(test_merger)
            log.info("Test data merged")

            if mutate_datasets is not None:
                train_iterable, val_iterable = mutate(
                    train_iterable,
                    val_iterable,
                    ensemble_index=mutate_datasets,
                    rseed=random_seed,
                )

            if store_se is True:
                energy_shifter = se

            # save to pkl
            with open(data_pkl_path, "wb") as fout:
                log.info("Saving to pickle {} .....".format(data_pkl_path))
                pickle.dump(
                    {
                        "training": train_iterable,
                        "test": test_iterable,
                        "validation": val_iterable,
                        "self_energies": energy_shifter.self_energies.cpu(),
                    },
                    fout,
                )

                log.info("Pickle saved")

            # Use iterators to save memory note the same data are used as pickled.
            log.info("Loading data into memory .....")
            train_merger = data_merge(*train_iterators)
            train_iterable = torchani.data.TransformableIterable(train_merger)

            val_merger = data_merge(*val_iterators)
            val_iterable = torchani.data.TransformableIterable(val_merger)

            test_merger = data_merge(*test_iterators)
            test_iterable = torchani.data.TransformableIterable(test_merger)

            train_data = train_iterable.collate(batch_size).cache()
            val_data = val_iterable.collate(batch_size).cache()
            test_data = test_iterable.collate(batch_size).cache()
            log.info("Data in memory ready fro training")

        else:
            if len(files) <= 0:
                log.error("No data (*.h5) files in {}".format(data))
                raise RuntimeError("No data *.h5 files present to load")

            log.info("Loading directly as only one data h5 file {}".format(files))

            if forces is False:
                log.info("Will not use forces data (--forces not used)")
                train, val, test = (
                    torchani.data.load(files[0])
                    .subtract_self_energies(se, species_order)
                    .species_to_indices(species_indicies)
                    .shuffle()
                    .split(train_size, val_size, None)
                )
            else:
                log.info("Will use forces data (--forces used)")
                train, val, test = (
                    torchani.data.load(files[0], additional_properties=("forces",))
                    .subtract_self_energies(se, species_order)
                    .species_to_indices(species_indicies)
                    .shuffle()
                    .split(train_size, val_size, None)
                )
            log.info("Training set size: {}".format(len(train)))
            log.info("validation set size: {}".format(len(val)))
            log.info("Test set size: {}".format(len(test)))
            train_iterable = torchani.data.TransformableIterable(train)
            log.info("Training data loaded into transformable iterable")

            val_iterable = torchani.data.TransformableIterable(val)
            log.info("Validation data loaded into transformable iterable")

            test_iterable = torchani.data.TransformableIterable(test)
            log.info("Test data loaded into transformable iterable")

            if mutate_datasets is not None:
                train_iterable, val_iterable = mutate(
                    train_iterable,
                    val_iterable,
                    ensemble_index=mutate_datasets,
                    rseed=random_seed,
                )

            if store_se is True:
                energy_shifter = se

            # save to pkl
            with open(data_pkl_path, "wb") as fout:
                log.info("Saving to pickle {} .....".format(data_pkl_path))
                pickle.dump(
                    {
                        "training": train_iterable,
                        "test": test_iterable,
                        "validation": val_iterable,
                        "self_energies": energy_shifter.self_energies.cpu(),
                    },
                    fout,
                )

                log.info("Pickle saved")

            train_data = train_iterable.collate(batch_size).cache()
            val_data = val_iterable.collate(batch_size).cache()
            test_data = test_iterable.collate(batch_size).cache()
            log.info("Data in memory ready fro training")

    log.info("Training set and validation set loaded")
    log.info("Self atomic energies: {}".format(energy_shifter.self_energies))

    return train_data, val_data, test_data


def mutate(
    train_data: torchani.data.TransformableIterable,
    val_data: torchani.data.TransformableIterable,
    ensemble_index: int = 0,
    rseed: int = 15234,
    min_fraction: float = 0.0,
    max_fraction: float = 0.1,
) -> (torchani.data.TransformableIterable, torchani.data.TransformableIterable):
    """
    Function to split and mix train and validation data using ensemble_index as a pseudo random seed
    :param train: torchani.data.TransformableIterable - training data as a torchani.data.TransformableIterable
    :param val: torchani.data.TransformableIterable - validation data as a torchani.data.TransformableIterable
    :param ensemble_index: int - seed for numpy random number
    :param rseed: int - numpy original random seed to reinitialize
    :param min_fraction: float - minimum fraction of entries to swap
    :param max_fraction: float - maximum fraction of entries to swap
    :return: (torchani.data.TransformableIterable, torchani.data.TransformableIterable) - training and validation data
    """

    log = logging.getLogger(__name__)

    np.random.seed(ensemble_index)
    fraction_to_change = np.random.uniform(low=min_fraction, high=max_fraction)
    log.info(f"Changing {fraction_to_change} fraction of train and validation data")

    train_min_index = int(
        np.floor(len(train_data.wrapped_iterable) * fraction_to_change)
    )
    if train_min_index != 0:
        train_max_index = train_min_index + train_min_index
    else:
        train_max_index = 1
    log.info(
        f"training index min {train_min_index} and max {train_max_index} index for swapping."
    )
    removed_training = train_data.wrapped_iterable[train_min_index:train_max_index]
    train_data = torchani.data.TransformableIterable(
        train_data.wrapped_iterable[:train_min_index]
        + train_data.wrapped_iterable[train_max_index:]
    )

    val_min_index = int(np.floor(len(val_data.wrapped_iterable) * fraction_to_change))
    if val_min_index != 0:
        val_max_index = val_min_index + val_min_index
    else:
        val_max_index = 1
    log.info(
        f"validation index min {val_min_index} and max {val_max_index} index for swapping."
    )
    removed_val = val_data.wrapped_iterable[val_min_index:val_max_index]
    val_data = torchani.data.TransformableIterable(
        val_data.wrapped_iterable[:val_min_index]
        + val_data.wrapped_iterable[val_max_index:]
    )

    train_data = torchani.data.TransformableIterable(
        train_data.wrapped_iterable + removed_val
    )
    val_data = torchani.data.TransformableIterable(
        val_data.wrapped_iterable + removed_training
    )

    np.random.seed(rseed)

    return train_data, val_data


def _original_load_ani_data(
    energy_shifter: torchani.utils.EnergyShifter,
    species_order: list,
    data: str,
    batch_size: int,
    train_size: float,
    val_size: float,
    forces: bool = False,
    no_reload: bool = False,
):

    """
    Function to load and merger ANI dataset
    :param energy_shifter: torchani.utils.EnergyShifter -
    :param species_order: itertable - list of element symbols ["H", "C", "N", "O"]
    :param data: str - path to find HDF5 files suitable for ani dataloader with extension *.h5
    :param batch_size: int - number of sample to train on each batch
    :param train_size: float - fraction of the data for training
    :param val_size: float - fraction of the data for validation
    :param forces: bool - use forces in the training for not
    :param no_reload: bool - use the previously stored data or reload
    :return: (torchani.data.TransformableIterable, torchani.data.TransformableIterable,
            torchani.data.TransformableIterable) - merged training, validation, and test data
    """

    log = logging.getLogger(__name__)

    train_iterators = []
    val_iterators = []
    test_iterators = []

    data_pkl_path = "ani_datasets.pkl"

    if no_reload is False:
        log.info("If {} is present will reload the file".format(data_pkl_path))
    else:
        log.info(
            "Will load data fresh and not use any previous stored data no_reload option given"
        )

    log.info("Batch size: {}".format(batch_size))

    if os.path.isfile(data_pkl_path) and no_reload is False:
        log.info(
            "Loading dataset to maintain consistent train, test, validation "
            "splitting"
        )

        with open(data_pkl_path, "rb") as fin:
            dataset = pickle.load(fin)

            try:
                train_data = dataset["training"].collate(batch_size).cache()
            except KeyError as kerr:
                log.error("ERROR - loading training data from {}".format(data_pkl_path))
                raise (kerr)

            try:
                test_data = dataset["test"].collate(batch_size).cache()
            except KeyError as kerr:
                log.error("ERROR - loading test data from {}".format(data_pkl_path))
                raise (kerr)

            try:
                val_data = dataset["validation"].collate(batch_size).cache()
            except KeyError as kerr:
                log.error(
                    "ERROR - loading validation data from {}".format(data_pkl_path)
                )
                raise (kerr)

            try:
                energy_shifter.self_energies = dataset["self_energies"]
            except KeyError as kerr:
                log.error(
                    "ERROR - loading self energy data from {}".format(data_pkl_path)
                )
                raise (kerr)

    else:
        # load data sets
        store_se = False
        if energy_shifter.self_energies is not None:
            log.info("The energy shifter provided is already initialized")
            log.info(
                "\n-----\nWill use the values of this initialized energy shifter rather than"
                "setting new ones for this data set.\nIf you want new one to be set"
                "pass in `torchani.EnergyShifter(None)`.\nWe assume that the"
                "energies are in the same order as species order, please check the output dictionary\n"
                "------\n"
            )
            se = {
                k: energy_shifter.self_energies[ith]
                for ith, k in enumerate(species_order)
            }
            log.info(
                "\n----- PLEASE CHECK ------\nPlease check this dictionary is correct:\n{}\n------".format(
                    se
                )
            )
        else:
            se = energy_shifter
            store_se = True

        files = sorted(glob.glob(os.path.join(data, "*.h5")))
        if len(files) > 1:
            for ith, finh5 in enumerate(files):
                log.info("loading {}".format(finh5))

                if forces is False:
                    log.info("Will not use forces data (--forces not used)")
                    training, validation, test = (
                        torchani.data.load(finh5)
                        .subtract_self_energies(se, species_order)
                        .species_to_indices(species_order)
                        .shuffle()
                        .split(train_size, val_size, None)
                    )
                else:
                    log.info("Will use forces data (--forces used)")
                    training, validation, test = (
                        torchani.data.load(finh5, additional_properties=("forces",))
                        .subtract_self_energies(se, species_order)
                        .species_to_indices(species_order)
                        .shuffle()
                        .split(train_size, val_size, None)
                    )

                log.info("{} loaded".format(finh5))
                train_iterators.append(training)
                val_iterators.append(validation)
                test_iterators.append(test)

            if any(
                len(ent) == 0
                for ent in [train_iterators, val_iterators, test_iterators]
            ):
                log.warning(
                    "WARNING - there appear to be no *.h5 files given as input in "
                    "{} "
                    "please check inputs.".format(data)
                )
                raise RuntimeError("No inputs")

            # Lists are needed not iterators so that the data can be pickeled
            # May be this leads to memory issues in which case we will need
            # to be careful here
            train_merger = list(data_merge(*train_iterators))
            train_iterable = torchani.data.TransformableIterable(train_merger)
            log.info("Training data merged")

            val_merger = list(data_merge(*val_iterators))
            val_iterable = torchani.data.TransformableIterable(val_merger)
            log.info("Validation data merged")

            test_merger = list(data_merge(*test_iterators))
            test_iterable = torchani.data.TransformableIterable(test_merger)
            log.info("Test data merged")

            # if store_se is True:
            #    energy_shifter = se

            # save to pkl
            with open(data_pkl_path, "wb") as fout:
                log.info("Saving to pickle .....")
                pickle.dump(
                    {
                        "training": train_iterable,
                        "test": test_iterable,
                        "validation": val_iterable,
                        "self_energies": se.self_energies.cpu(),
                    },
                    fout,
                )

                log.info("Pickle saved")

            # Use iterators to save memory note the same data are used as pickled.
            log.info("Loading data into memory .....")
            train_merger = data_merge(*train_iterators)
            train_iterable = torchani.data.TransformableIterable(train_merger)

            val_merger = data_merge(*val_iterators)
            val_iterable = torchani.data.TransformableIterable(val_merger)

            test_merger = data_merge(*test_iterators)
            test_iterable = torchani.data.TransformableIterable(test_merger)

            train_data = train_iterable.collate(batch_size).cache()
            val_data = val_iterable.collate(batch_size).cache()
            test_data = test_iterable.collate(batch_size).cache()
            log.info("Data in memory ready fro training")

        else:
            if len(files) <= 0:
                log.error("No data (*.h5) files in {}".format(data))
                raise RuntimeError("No data *.h5 files present to load")
            log.info("Loading directly as only one data h5 file")
            train, val, test = (
                torchani.data.load(files[0])
                .subtract_self_energies(energy_shifter, species_order)
                .species_to_indices(species_order)
                .shuffle()
                .split(train_size, None)
            )

            train_iterable = torchani.data.TransformableIterable(train)
            val_iterable = torchani.data.TransformableIterable(val)
            test_iterable = torchani.data.TransformableIterable(test)

            # save to pkl
            log.info("Data loaded saving to pickle file")
            with open(data_pkl_path, "wb") as fout:
                log.info("Saving to pickle .....")
                pickle.dump(
                    {
                        "training": train_iterable,
                        "test": test_iterable,
                        "validation": val_iterable,
                        "self_energies": se.self_energies.cpu(),
                    },
                    fout,
                )

                log.info("Pickle saved")

            train_data = train_iterable.collate(batch_size).cache()
            val_data = val_iterable.collate(batch_size).cache()
            test_data = test_iterable.collate(batch_size).cache()

    log.info("Training set, validation set and test set loaded")
    log.info("Self atomic energies: {}".format(energy_shifter.self_energies))

    return train_data, val_data, test_data
