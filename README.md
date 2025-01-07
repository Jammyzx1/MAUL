# MAUL (Modified ANI with Uncertainity Limits)
Maul is a set of Bayesian neural networks built based upon the DNN ensemble ANI. This
library contains the key tools to build and train a Maul neural potential. The neural 
potential offers a predicted energy together with an estimate of the model (epistemic) 
uncertaininty based upon Monte Carlo sampling over parameter distributions.

## Library
The library consists of five core modules built on top of [pytorch](https://pytorch.org/), 
[torchani](https://github.com/aiqm/torchani) and [Bayesian torch](https://github.com/IntelLabs/bayesian-torch).
The core modules are:

1. network\_classes.py
1. network\_data\_loader.py
1. network\_train.py
1. network\_test.py
1. network\_utilities.py

Firstly, `network_classes.py` defines a super class for ANI style networks, enabling efficient
automated building of these networks and methods to track their status. The core class here
is the `ANI()` class. From this module one can define a standard ANI potential and in one 
command convert the potential to Maul style BNN neural potential.

Next, we have `newtork_data_loader.py`. This module as it says is for loading data in a consistent way.
The module is designed around the HDF5 files from the ANI1 and ANI-1x datasets. The module allows the data used
to be mutated in a consistent way by passing a `--mutate_index INT` which becomes a seed to repeatably partition the 
trianing and validation sets such that over an ensemble each member has a different sub set.

`Network_train.py` and `network_test.py` have functionality for training and testing the neural potentials
The modules can train and test DNN neural potentials and BNN neural potentials. 

Finally, `network_utilities.py` is an important module conatining most of the plotting tools for analysis of performance.

## Install

Build a new environment using conda or venv

`python -m venv`

or

`conda create -n maul-bnnani python=3.9`

`conda activate maul-bnnani`

Install the Maul bnnani library

`python setup.py install`

## Args

optional arguments:

  -h, --help            show this help message and exit

Featurization options:

  --radial\_cutoff RADIAL\_CUTOFF
                        Radial cutoff distance (default: 5.2)

  --theta\_max\_rad THETA\_MAX\_RAD
                        Theta distance cutoff (default: 3.33794218)

  --angular\_cutoff ANGULAR\_CUTOFF
                        Angular cutoff in radians (default: 3.5)

  --etar ETAR           Eta in the radial function determines the guassians
                        width (default: 16.0)

  --etaa ETAA           Eta in the angular function determines the guassians
                        width for the radial portion (default: 8.0)

  --zeta ZETA           Zeta in the angular function determines the guassians
                        width for the angular portion (default: 32.0)

  --radial\_steps RADIAL\_STEPS
                        Number of radial shifts moves the peak of the gaussian
                        (default: 16)

  --angular\_radial\_steps ANGULAR\_RADIAL\_STEPS
                        Number of angular shifts moves the peak of the
                        gaussian (default: 4)

  --theta\_steps THETA\_STEPS
                        Angle steps (default: 8)

  --species\_order [SPECIES\_ORDER [SPECIES\_ORDER ...]]
                        Species order at atomic symbol (default: ['H', 'C',
                        'N', 'O'])

  --self\_energies [SELF\_ENERGIES [SELF\_ENERGIES ...]]
                        Self energies in the order of species order (default:
                        None)

Data options:

  --data str            Data directory where HDF5 file of training and testing
                        data can be found (default: ../../data/ANI-1\_release)

  --no\_reload           Tell the code to reload data don't use previous
                        loading data (default: False)

  --checkpoint\_filename CHECKPOINT\_FILENAME
                        initial weights path (default:
                        ani\_model\_checkpoint.pt)

  --learning\_curve\_csv\_filename LEARNING\_CURVE\_CSV\_FILENAME
                        best weights path (default: learning\_curve.csv)

  --batch\_size BATCH\_SIZE
                        Batch size for training (default: 1024)

  --learning\_rate LEARNING\_RATE
                        Learning rate (default: 1e-06)

  --train\_size TRAIN\_SIZE
                        Training dataset size (default: 0.8)

  --validation\_size VALIDATION\_SIZE
                        Validation dataset size (default: 0.1)

  --random\_seed RANDOM\_SEED
                        random seed tp initialize (default: 15234)

  --forces              Train set has forces and use them (default: False)

  --force\_scalar FORCE\_SCALAR
                        the weight to apply to the forces part of the loss
                        when training (default: 1.0)

  --max\_epochs MAX\_EPOCHS
                        maximum number of epochs (default: 1000)

  --species\_indicies SPECIES\_INDICIES [SPECIES\_INDICIES ...]
                        Whether to read species by atomic number (ani 1x data)
                        or label mapped to network index (ani 1 data)
                        (default: ['periodic\_table'])

  --early\_stopping\_learning\_rate EARLY\_STOPPING\_LEARNING\_RATE
                        Early stopping if this learning rate is met (default:
                        1e-09)

BNN prior options:

  --params\_prior PARAMS\_PRIOR
                        'ani' or Json file with prior parameters stored under
                        appropriate keys for setting each individually
                        (default: None)

  --prior\_mu PRIOR\_MU   Prior mean (default: 0.0)

  --prior\_sigma PRIOR\_SIGMA
                        Prior standard deviation (default: 1.0)

  --posterior\_mu\_init POSTERIOR\_MU\_INIT
                        Posterior mean initialization (default: 0.0)

  --posterior\_rho\_init POSTERIOR\_RHO\_INIT
                        Posterior denisty initialization (default: -3.0)

  --reparameterization  Use the reparameterization version of the bayesian
                        layers (default: False)

  --flipout             Use the flipout version of the bayesian layers
                        (default: False)

  --moped               Use the moped priors (default: False)

  --moped\_delta MOPED\_DELTA
                        MOPED delta value (default: 0.2)

  --moped\_enable        Initialize mu/sigma from the dnn weights (default:
                        False)

  --moped\_init\_model MOPED\_INIT\_MODEL
                        DNN model to intialize MOPED method filename and path
                        /path/to/dnn\_v1\_moped\_init.pt (default:
                        dnn\_v1\_moped\_init.pt)

  --bayesian\_depth BAYESIAN\_DEPTH
                        Use bayesian layers to a depth of n (None means all)
                        (default: None)

  --lml                 Whether to replace the MSE loss with LML loss for the
                        BNN (default: False)

Training options:

  --use\_schedulers      Use lr schedulers for adam and sgd (default: False)

  --train\_mc\_runs TRAIN\_MC\_RUNS
                        Number of Monte Carlo runs during training of the
                        Bayesian potential (default: 2)

  --test\_mc\_runs TEST\_MC\_RUNS
                        Number of Monte Carlo runs during testing of the
                        Bayesian potential (default: 2)

  --kl\_weight KL\_WEIGHT
                        KL factor (default: 0.0)

  --load\_pretrained\_parameters\_dnn LOAD\_PRETRAINED\_PARAMETERS\_DNN
                        Path and file name to load pre-trained DNN parameters
                        from (default: None)

  --load\_pretrained\_parameters\_bnn LOAD\_PRETRAINED\_PARAMETERS\_BNN
                        Path and file name to load pre-trained BNN parameters
                        from (default: None)

  --pretrained\_model\_key PRETRAINED\_MODEL\_KEY
                        if there is a key the model weights and biases are
                        stored give it here (default: None)

  --update\_bnn\_variance UPDATE\_BNN\_VARIANCE
                        If you load a pretrained BNN but want to update the
                        varience of the parameters set a fraction between 0.0
                        and 1.0. (default: None)

  --name NAME           Name for the model sor reference later on (default:
                        my)

  --mutate\_index MUTATE\_INDEX
                        Permute dataset swapping examples in training and
                        validation set using this index as a randomseed. None
                        will not permute the data set beyond the normal
                        loading (default: None)

  --no\_mu\_opt           Whether to omit training the mean weights of the BNN
                        (default: False)

  --no\_sig\_opt          Whether to omit training the sigma weights of the BNN
                        (default: False)

Running options:

  --train\_ani\_dnn       Train a DNN (default: False)

  --train\_ani\_bnn       Train a BNN (default: False)

  --ensemble ENSEMBLE   Number of concurrent models to train as an ensemble
                        prediction (default: 1)

  --set\_rho\_explicit    If params prior is being used will set rho values
                        using the params prior (default: False)

  --set\_prior\_explicit  If params prior is being used will set prior values
                        using the params prior (default: False)

  --use\_cuaev           Use the cuda ave codes note these are installed
                        separately from ani library (default: False)

  --summary\_plot\_only   in the training only output summary plots for each
                        epoch (default: True)

  --parallel            Run using data parallel pytorch stratergy (default:
                        False)

Logging arguments:
  --loglevel LOGLEVEL   log level (default: INFO)

