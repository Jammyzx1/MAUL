import torch
import torchani
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Any, NoReturn, Union
import re
from src.modified_ase import Calculator
from ase import Atoms, units
from ase.optimize import BFGS
from ase.io import write
from ase.md import MDLogger
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import shutil

def run(mol_name, positions, bnn = False, optimise = True, md = False):
    species = ["H", "C", "N", "O"]

    if bnn == False:
        model = torch.load("complete_models/ensemble.pt")
        mol = Atoms(mol_name,
                    positions=positions)
        mol.calc = Calculator(species=species, model=model, mc_runs = 1)
        if optimise == True:
            opt = BFGS(mol, logfile = "dnn.log")
            opt.run(fmax = 0.02)
            mol.get_forces()
        if md == True:
            MaxwellBoltzmannDistribution(mol, temperature_K=273)
            dyn = NVTBerendsen(mol, 0.1 * units.fs, 273, taut=0.5*1000*units.fs, trajectory='md_nvt_dnn.traj', logfile='md_nvt_dnn.log')
            dyn.run(20000)
        mol.get_total_energy()
        try:
            dnn_forces = mol.calc.results["forces"]
        except KeyError:
            dnn_forces = 0

        return mol.calc.results["free_energy"], mol.calc.results["free_energy_uncertainty"], dnn_forces

    if bnn == True:
        model = torch.load("complete_models/maul1.pt")
        mol = Atoms(mol_name,
                    positions=positions)
        mol.calc = Calculator(species=species, model=model, mc_runs = 20)
        if optimise == True:
            opt = BFGS(mol, logfile = "bnn.log")
            opt.run(fmax=0.5)
        if md == True:
            MaxwellBoltzmannDistribution(mol, temperature_K=273)
            dyn = NVTBerendsen(mol, 0.1 * units.fs, 273, taut=0.5*1000*units.fs, trajectory='md_nvt_bnn.traj', logfile='md_nvt_bnn.log')
            dyn.run(20000)
        mol.get_total_energy()

        bnn_en = mol.calc.results["free_energy"]
        bnn_en_unc = mol.calc.results["free_energy_uncertainty"]
        try:
            bnn_forces = mol.calc.results["forces"]
        except KeyError:
            bnn_forces = 0

        return bnn_en, bnn_en_unc, bnn_forces

if __name__ == '__main__':

    # Example Script
    json_fn =  "complete_models/all_compounds_data.json"
    with open(json_fn) as json_data:
        d = json.load(json_data)
    dat = pd.DataFrame(d)

    index = 1
    mol_name = dat["symbols"][index]
    positions = dat['positions'][index]
    ener1, std1, fr1 = run(mol_name, positions, bnn = True, optimise = True, md = False)
