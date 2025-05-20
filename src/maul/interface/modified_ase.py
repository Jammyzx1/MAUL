# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.
.. _ASE:
    https://wiki.fysik.dtu.dk/ase


Base for the code from https://github.com/aiqm/torchani/blob/master/torchani/ase.py provided under MIT license
https://github.com/aiqm/torchani/blob/master/LICENSE 20/9/2022
"""

import torch
from torchani import utils
import ase.calculators.calculator
import ase.units
import re
import numpy as np
from maul.neural_pot.maul_library.network_classes import ANI, bnn_priors
from maul.neural_pot.maul_library.network_utilities import (
    get_ani_parameter_distrbutions,
)
from itertools import groupby


class Calculator(ase.calculators.calculator.Calculator):
    """BNN-ANI calculator.py for ASE
    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
        model (:class:`torch.nn.Module`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
        mc_runs (integer): Number of samples to draw to determine energy mean/uncertainty
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "free_energy",
        "energy_uncertainty",
        "free_energy_uncertainty",
    ]

    def __init__(
        self,
        species,
        model,
        overwrite=False,
        mc_runs=20,
        mc_forces_cov=False,
        periodic_table_indices=False,
        units="eV",
    ):
        """
        Initialization funtion for the class
        :param species: sequence of strings or string - element labels
        :param model: torch.nn.Module - torch model to run the evaluations
        :param overwrite: bool - overwrite or not
        :param mc_runs: int - number of monte carlo runs to use for sampling
        :param mc_forces_cov: bool - flag indicating whether to compute the full covariance for the forces
        :param periodic_table_indices: bool - use periodic table numbering or not
        :param units: str - expected units has no direct effect but makes it easy to know the units you are working in
        """
        super().__init__()
        self.model = model
        self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
        self.mc_runs = mc_runs
        self.mc_forces_cov = mc_forces_cov
        self.units = units
        # Since ANI is used in inference mode, no gradients on model parameters are required here
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.overwrite = overwrite

        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype

        if periodic_table_indices is True:
            # We assume that the model has a "periodic_table_index" attribute
            # if it doesn't we set the calculator.py's attribute to false and we
            # assume that species will be correctly transformed by
            # species_to_tensor
            self.periodic_table_index = periodic_table_indices
        else:
            self.periodic_table_index = False

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=["energy", "energy_uncertainty"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        """
        Method to calculate energies, stresses and forces
        :param atoms: ase.Atoms or None - the atoms object to use
        :param properties: list - properties to compute
        :param system_changes: ase.calculators.calculator.all_changes - updated changes to system
        :return:
        """
        super().calculate(atoms, properties, system_changes)
        cell = torch.tensor(
            np.array(self.atoms.get_cell(complete=True)),
            dtype=self.dtype,
            device=self.device,
        )
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool, device=self.device)
        pbc_enabled = pbc.any().item()

        if self.periodic_table_index:
            species = torch.tensor(
                self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
            )
        else:
            species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(
                self.device
            )

        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = (
            coordinates.to(self.device)
            .to(self.dtype)
            .requires_grad_("forces" in properties)
        )

        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
            if self.overwrite and atoms is not None:
                atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if "stress" in properties:
            scaling = torch.eye(
                3, requires_grad=True, dtype=self.dtype, device=self.device
            )
            coordinates = coordinates @ scaling
        coordinates = coordinates

        sp_stack = torch.stack([species for _ in range(self.mc_runs)])
        coord_stack = torch.stack([coordinates for _ in range(self.mc_runs)])

        if pbc_enabled:
            if "stress" in properties:
                cell = cell @ scaling
            _, energies = self.model((sp_stack, coord_stack), cell=cell, pbc=pbc)
        else:
            specs, energies = self.model((sp_stack, coord_stack))

        energies *= ase.units.Hartree

        energy = energies.mean()
        energy_std = energies.std()

        self.results["energy"] = energy.item()
        self.results["free_energy"] = energy.item()
        self.results["energy_uncertainty"] = energy_std.item()
        self.results["free_energy_uncertainty"] = energy_std.item()

        if "forces" in properties:
            n_atoms = coordinates.shape[0]
            grad = torch.autograd.grad(
                energies,
                coordinates,
                grad_outputs=torch.diag(torch.ones(self.mc_runs)),
                retain_graph="stress" in properties,
                is_grads_batched=True,
            )
            forces = -grad[0].mean(axis=0)

            ravelled_forces = -grad[0].reshape([self.mc_runs, -1])
            if self.mc_forces_cov:
                ravelled_forces_cov = ravelled_forces.t().cov()
                forces_std = torch.diag(ravelled_forces_cov).reshape(n_atoms, 3) ** 0.5
                forces_cov = ravelled_forces_cov.reshape(n_atoms, 3, n_atoms, 3)
            else:
                forces_std = ravelled_forces.std(axis=0).reshape(n_atoms, 3)
                forces_cov = forces_std**2

            self.results["forces"] = forces.squeeze(0).to("cpu").numpy()
            self.results["forces_uncertainty"] = forces_std.to("cpu").numpy()
            self.results["forces_covariance"] = forces_cov.to("cpu").numpy()

        if "stress" in properties:
            volume = self.atoms.get_volume()
            grad = torch.autograd.grad(
                energies,
                scaling,
                grad_outputs=torch.diag(torch.ones(self.mc_runs)),
                is_grads_batched=True,
            )
            stress = grad[0].mean(axis=0) / volume
            stress_std = grad[0].std(axis=0)

            self.results["stress"] = stress.cpu().numpy()
            self.results["stress_uncertainty"] = stress_std.cpu().numpy()

    def calculate_many(
        self,
        list_atoms,
        max_block_size=None,
        properties=["energies", "energy_uncertainties"],
    ):
        """
        Method to calculate energies, stresses and forces for multiple atoms objets
        :param list_atoms: list of atoms objects to compute properties for
        :param properties: list - properties to compute
        :return: results: dict - calculated properties
        """
        if "stress" in properties:
            raise RuntimeError("Stress not yet supported within calculate_many")

        if self.mc_forces_cov:
            raise RuntimeError(
                "Full Force Covariance calculation not supported within calculate_many"
            )

        sorted_ind_atoms = sorted(enumerate(list_atoms), key=lambda a: len(a[1]))
        group_es, group_stds, group_forces, group_fstds, group_fcov, group_inds = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for size, ind_mols in groupby(sorted_ind_atoms, lambda e: len(e[1])):
            inds, mols = zip(*ind_mols)

            pbc = (
                torch.from_numpy(np.array([m.get_pbc() for m in mols]))
                .to(torch.bool)
                .to(self.device)
            )

            # assume if any pbc then pbc applies to all
            if pbc.any().item():
                raise RuntimeError("PBC not yet supported within calculate_many")

            if self.periodic_table_index:
                species = (
                    torch.from_numpy(np.array([m.get_atomic_numbers() for m in mols]))
                    .to(torch.long)
                    .to(self.device)
                )
            else:
                species = torch.from_numpy(
                    np.array([m.get_chemical_symbols() for m in mols])
                ).to(self.device)

            coordinates = torch.from_numpy(np.array([m.get_positions() for m in mols]))
            coordinates = (
                coordinates.to(self.device)
                .to(self.dtype)
                .requires_grad_("forces" in properties)
            )

            sp_stack = torch.vstack([species for _ in range(self.mc_runs)])
            coord_stack = torch.vstack([coordinates for _ in range(self.mc_runs)])

            specs, energies = self.model((sp_stack, coord_stack))

            energies *= ase.units.Hartree

            # reshape to make the mc_runs repeats the first dimension
            renergies = energies.reshape([self.mc_runs, -1])
            energy = renergies.mean(axis=0)
            energy_std = renergies.std(axis=0)

            group_es.append(energy)
            group_stds.append(energy_std)
            group_inds.append(torch.Tensor(inds).to(torch.long))

            if "forces" in properties:
                n_mols = coordinates.shape[0]
                n_atoms = coordinates.shape[1]
                grad = torch.autograd.grad(
                    energies,
                    coordinates,
                    grad_outputs=torch.diag(torch.ones(n_mols * self.mc_runs)),
                    retain_graph="stress" in properties,
                    is_grads_batched=True,
                )
                # gradient calculated for all energies and all coordinates, i.e. we compute derivatives of the energy
                # of mol1 with respective to the coordinates of all the molecules. To remove the blocks of zero gradient
                # we create and apply a  mask
                # TODO check whether the speed up from block evaluating the gradients outweights the generation of all
                # there zero elements - the alterrnative is to loop over the the coordinates and call grad n_mol times
                mask = torch.diag(torch.ones(n_mols, device=self.device)).to(torch.bool)
                grads = grad[0].reshape(self.mc_runs, n_mols, n_mols, n_atoms, 3)[
                    :, mask, :
                ]

                forces = -grads.mean(axis=0)
                ravelled_forces = -grads.reshape([self.mc_runs, n_mols, -1])

                forces_std = ravelled_forces.std(axis=0).reshape(n_mols, n_atoms, 3)
                forces_cov = forces_std**2

                group_forces.append(forces)
                group_fstds.append(forces_std)
                group_fcov.append(forces_cov)

        group_es = torch.hstack(group_es).ravel()
        group_stds = torch.hstack(group_stds).ravel()
        group_inds = torch.hstack(group_inds).ravel()

        master_energies = torch.zeros_like(group_es)
        master_stds = torch.zeros_like(group_stds)

        master_energies[group_inds] = group_es
        master_stds[group_inds] = group_stds

        self.results["energies"] = master_energies.detach().cpu().numpy()
        self.results["energy_uncertainties"] = master_stds.detach().cpu().numpy()

        if "forces" in properties:
            master_inds = np.argsort(group_inds)
            ungrouped_forces = [f for g in group_forces for f in g.cpu().numpy()]
            ungrouped_fstds = [s for g in group_fstds for s in g.cpu().numpy()]
            ungrouped_fcov = [c for g in group_fcov for c in g.cpu().numpy()]

            master_forces = [ungrouped_forces[i] for i in master_inds]
            master_fstds = [ungrouped_fstds[i] for i in master_inds]
            master_fcov = [ungrouped_fcov[i] for i in master_inds]

            self.results["forces"] = master_forces
            self.results["force_uncertainties"] = master_fstds
            self.results["force_covariances"] = master_fcov

        return self.results


def load_model(checkpoint_fn: str, model_key: str = "nn", default=None, bnn=True):
    """
    Function to load the BNN ANI model from a pytorch (*.pt)
    :param checkpoint_fn: file name to load the model from
    :param model key: key for the pt file to load the neural networks only
    :param default: dict - DO NOT CHANGE IN GENERAL only change if your model is different
    :return: ANI model and species
    """
    if default is None:
        default = {
            "forces": False,
            "force_scalar": 1.0,
            "self_energies": None,
            "radial_cutoff": 5.2000e00,
            "theta_max_rad": 3.33794218e00,
            "angular_cutoff": 3.5000e00,
            "etar": 1.6000000e01,
            "etaa": 8.00000,
            "zeta": 3.2000000e01,
            "radial_steps": 16,
            "angular_radial_steps": 4,
            "theta_steps": 8,
            "species_order": ["H", "C", "N", "O"],
            "species_indicies": "periodic_table",
            "ensemble": 1,
            "use_cuaev": False,
            "reparameterization": True,
            "flipout": False,
            "prior_mu": 0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0,
            "posterior_rho_init": 3.0,
            "moped_enable": False,
            "moped_delta": 0.2,
            "set_rho_explicit": False,
        }

        default["self_energies"] = [-0.6010, -38.0832, -54.7078, -75.1945]
        default["reparameterization"] = False
        default["flipout"] = True
        default["moped_enable"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ani_class = ANI(
        forces=default["forces"],
        force_scalar=default["force_scalar"],
        self_energies=default["self_energies"],
    )

    ani_class.build_ani_dnn_model(
        default["radial_cutoff"],
        default["theta_max_rad"],
        default["angular_cutoff"],
        default["etar"],
        default["etaa"],
        default["zeta"],
        default["radial_steps"],
        default["angular_radial_steps"],
        default["theta_steps"],
        default["species_order"],
        networks=None,
        ensemble=default["ensemble"],
        use_cuaev=default["use_cuaev"],
        no_species_converter=False
        if default["species_indicies"] == "periodic_table"
        else True,
    )

    bayesian_priors = bnn_priors(
        default["reparameterization"],
        default["flipout"],
        default["prior_mu"],
        default["prior_sigma"],
        default["posterior_mu_init"],
        default["posterior_rho_init"],
        default["moped_enable"],
        default["moped_delta"],
    )

    # TODO: this is not a great way to do this as we may not have an ensemble at some point we should find a better
    #  way to handle this
    for il, layer in enumerate(ani_class.model):
        if re.search(r"^Ensemble", str(layer)):
            network_layer_number = il
    params_prior = get_ani_parameter_distrbutions(
        species_order=default["species_order"],
        prepend_key="{}.0.".format(network_layer_number),
    )

    if bnn:
        ani_class.transfer_dnn_to_bnn(
            bnn_prior=bayesian_priors,
            params_prior=params_prior,
            set_rho_explicitly=default["set_rho_explicit"],
            set_prior_explicitly=default["set_rho_explicit"],
        )

    ani_class.load_pretrained_on_to_model(checkpoint_fn, model_key=model_key)
    ani_class.complete_model()

    return ani_class.model, default["species_order"]


class FakeModel:
    def parameters(self):
        for i in range(10):
            yield torch.Tensor([i])

    def __call__(self, inp):
        return None, torch.Tensor([0.5])


def test():
    """
    Tests the model we built against ANI-1x directly, ANI-1x using the ase and Psi4 using the ase
    :return: None
    """
    import os
    import torchani

    # ANI-1x direct in this section is in Hartree
    ani1x = torchani.models.ANI1x(periodic_table_index=True)
    ani1x_en = ani1x(
        (
            torch.tensor([[1, 1]], requires_grad=False),
            torch.tensor([[[0, 0, 0], [0, 0, 0.75]]], requires_grad=True),
        )
    ).energies
    print("ANI-1x {}".format(torchani.units.hartree2ev(ani1x_en.item())))

    # ANI-1x ase energy in eV
    ani1x_ase_h2 = ase.Atoms(
        symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]
    )
    ani1x_ase_h2.calc = torchani.models.ANI1x(periodic_table_index=True).ase()
    ani1x_ase = ani1x_ase_h2.get_potential_energy()
    print("ANI-1x ase {} eV".format(ani1x_ase))

    # BNN ANI (MAUL Modified ANI with Uncertainity Limits to keep the star wars theme?) ase energy in eV
    checkpoint_fn = os.path.join(os.getcwd(), "ref_ext_ani_model_checkpoint.pt")

    # need to get the actual model from the checkpoint file
    model, species = load_model(checkpoint_fn)

    maul_h2 = ase.Atoms(
        symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]
    )
    maul_h2.calc = Calculator(species=species, model=model, mc_runs=100)
    maul_en = maul_h2.get_potential_energy()
    print(
        "BNN ANI : {} +/- {} eV".format(
            maul_en, maul_h2.calc.results["energy_uncertainty"]
        )
    )

    # Psi4 energy in eV
    try:
        from ase.calculators.psi4 import Psi4

        h2 = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])

        calPsi4Calculator = Psi4(
            atoms=h2,
            method="wB97x",
            memory="4000MB",  # 500MB is the default
            basis="6-31G*",
        )

        calPsi4Calculator.psi4.set_options(
            {
                "e_convergence": 1e-6,
                "d_convergence": 1e-6,
                "dft_radial_points": 99,
                "dft_spherical_points": 590,
                "dft_pruning_scheme": "robust",
            }
        )
        h2.calc = calPsi4Calculator
        h2_psi4_energy_ev = h2.get_potential_energy()
        print("ASE Psi4 {} eV".format(h2_psi4_energy_ev))

        print("\nComparison values in eV\n")
        print(" ani-1x  | ani-1x-ase | bnn-ani-ase | psi4-ase")
        print("---------|------------|-------------|---------")
        print(
            "{:.4f} |  {:.4f}  |   {:.4f}  | {:.4f}".format(
                torchani.units.hartree2ev(ani1x_en.item()),
                ani1x_ase,
                maul_en,
                h2_psi4_energy_ev,
            )
        )
    except ImportError as ierr:
        print("Psi4 no avaliable for direct comparison will use stored energy")

        print("\nComparison values in eV\n")
        print(" ani-1x  | ani-1x-ase | bnn-ani-ase | psi4-ase")
        print("---------|------------|-------------|---------")
        print(
            "{:.4f} |  {:.4f}  |   {:.4f}  | {:.4f}".format(
                torchani.units.hartree2ev(ani1x_en.item()),
                ani1x_ase,
                maul_en,
                -31.734216329568298,
            )
        )


def tests():
    """
    Test the model we have against psi4 for H2 and C6H6
    :return: None
    """
    import os
    import torchani
    from ase import Atoms

    checkpoint_fn = os.path.join(os.getcwd(), "ref_ext_ani_model_checkpoint.pt")

    # need to get the actual model from the checkpoint file
    model, species = load_model(checkpoint_fn)

    print(model)

    # model, species = FakeModel(), ['H', 'N' 'C', 'O']
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])

    h2.calc = Calculator(species=species, model=model, mc_runs=20)
    h2.get_total_energy()

    ani1x_ase_h2 = ase.Atoms(
        symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]
    )
    ani1x_ase_h2.calc = torchani.models.ANI1x(periodic_table_index=True).ase()
    ani1x_ase = ani1x_ase_h2.get_potential_energy()

    print("\nH2\n----")
    print("Psi4 calculation give -31.734216329568298 eV")
    print("ANI-1x ase {} eV".format(ani1x_ase))
    print(
        "BNNANI:",
        h2.calc.results["energy"],
        "+/-",
        h2.calc.results["energy_uncertainty"],
        "eV",
    )
    print(
        "BNNANI:",
        h2.calc.results["energy"] * 23.06,
        "+/-",
        h2.calc.results["energy_uncertainty"] * 23.06,
        "kcal/mol\n-----",
    )

    ring_species = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]

    ring_coordinates = [
        [-1.2131, -0.6884, 0.0000],
        [-1.2028, 0.7064, 0.0001],
        [-0.0103, -1.3948, 0.0000],
        [0.0104, 1.3948, -0.0001],
        [1.2028, -0.7063, 0.0000],
        [1.2131, 0.6884, 0.0000],
        [-2.1577, -1.2244, 0.0000],
        [-2.1393, 1.2564, 0.0001],
        [-0.0184, -2.4809, -0.0001],
        [0.0184, 2.4808, 0.0000],
        [2.1394, -1.2563, 0.0001],
        [2.1577, 1.2245, 0.0000],
    ]

    c6h6 = Atoms("C6H6", positions=ring_coordinates)

    c6h6.calc = Calculator(species=species, model=model, mc_runs=20, mc_forces_cov=True)
    c6h6.get_forces()
    c6h6.get_total_energy()

    ani1x_ase_c6h6 = ase.Atoms(symbols=ring_species, positions=ring_coordinates)
    ani1x_ase_c6h6.calc = torchani.models.ANI1x(periodic_table_index=True).ase()
    ani1x_ase = ani1x_ase_c6h6.get_potential_energy()

    print("\nC6H6 Benzene\n-----")
    print("Psi4 calculation give -6317.907899 eV")
    print("ANI-1x ase {} eV".format(ani1x_ase))
    print(
        "BNNANI:",
        c6h6.calc.results["energy"],
        "+/-",
        c6h6.calc.results["energy_uncertainty"],
        "eV",
    )
    print(
        "BNNANI:",
        c6h6.calc.results["energy"] * 23.06,
        "+/-",
        c6h6.calc.results["energy_uncertainty"] * 23.06,
        "kcal/mol\n-----",
    )

    print("BNNANI: force", c6h6.calc.results["forces"])
    print("BNNANI: force uncertainty", c6h6.calc.results["forces_uncertainty"])
    print("BNNANI: force cov", c6h6.calc.results["forces_covariance"])


def test_many():
    from ase import Atoms
    import os
    import torch

    checkpoint_fn = os.path.join(os.getcwd(), "ref_ext_ani_model_checkpoint.pt")

    # need to get the actual model from the checkpoint file
    model, species = load_model(checkpoint_fn)

    # model, species = FakeModel(), ['H', 'N' 'C', 'O']
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])

    ring_coordinates = [
        [-1.2131, -0.6884, 0.0000],
        [-1.2028, 0.7064, 0.0001],
        [-0.0103, -1.3948, 0.0000],
        [0.0104, 1.3948, -0.0001],
        [1.2028, -0.7063, 0.0000],
        [1.2131, 0.6884, 0.0000],
        [-2.1577, -1.2244, 0.0000],
        [-2.1393, 1.2564, 0.0001],
        [-0.0184, -2.4809, -0.0001],
        [0.0184, 2.4808, 0.0000],
        [2.1394, -1.2563, 0.0001],
        [2.1577, 1.2245, 0.0000],
    ]

    c6h6 = Atoms("C6H6", positions=ring_coordinates)

    calc = Calculator(species=species, model=model, mc_runs=20)

    torch.manual_seed(0)
    many_results = calc.calculate_many([h2, c6h6, h2], properties=["forces"])

    h2.calc = Calculator(species=species, model=model, mc_runs=20)
    torch.manual_seed(0)
    h2.get_forces()
    c6h6.calc = Calculator(species=species, model=model, mc_runs=20, mc_forces_cov=True)
    torch.manual_seed(0)
    c6h6.get_forces()

    # can't really account for different random seeds for the c6h6
    print(h2.calc.results["energy"] - many_results["energies"][0])
    print(c6h6.calc.results["energy"] - many_results["energies"][1])

    print(h2.calc.results["forces"] - many_results["forces"][0])
    print(c6h6.calc.results["forces"] - many_results["forces"][1])

    print(
        h2.calc.results["forces_uncertainty"] - many_results["force_uncertainties"][0]
    )
    print(
        c6h6.calc.results["forces_uncertainty"] - many_results["force_uncertainties"][1]
    )


def test_geom_opt():
    from ase import Atoms
    import os
    from ase.optimize import BFGS

    checkpoint_fn = os.path.join(os.getcwd(), "ref_bnn_ani_checkpoint.pt")

    # need to get the actual model from the checkpoint file
    model, species = load_model(checkpoint_fn)
    ring_coordinates = [
        [-1.2131, -0.6884, 0.0000],
        [-1.2028, 0.7064, 0.0001],
        [-0.0103, -1.3948, 0.0000],
        [0.0104, 1.3948, -0.0001],
        [1.2028, -0.7063, 0.0000],
        [1.2131, 0.6884, 0.0000],
        [-2.1577, -1.2244, 0.0000],
        [-2.1393, 1.2564, 0.0001],
        [-0.0184, -2.4809, -0.0001],
        [0.0184, 2.4808, 0.0000],
        [2.1394, -1.2563, 0.0001],
        [2.1577, 1.2245, 0.0000],
    ]

    c6h6 = Atoms("C6H6", positions=ring_coordinates)

    c6h6.calc = Calculator(species=species, model=model, mc_runs=20)

    opt = BFGS(c6h6, trajectory="opt.traj", logfile="opt.log")
    opt.run(fmax=0.05)


if __name__ == "__main__":
    # test()
    # tests()
    # test_many()
    test_geom_opt()
