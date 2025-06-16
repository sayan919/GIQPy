from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

CoordType = np.ndarray
AtomListType = List[str]
ChargeListType = List[float]
SolventGroupType = Tuple[AtomListType, CoordType]
MMChargeTupleType = Tuple[float, float, float, float]

@dataclass
class SolventCharge:
    element: str
    charge: float

@dataclass
class MonomerMeta:
    name: str
    nAtoms: int
    charge: float
    spin_mult: int
    mol_formula: str = ""

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class SolventMeta:
    name: str
    mol_formula: str
    nAtoms: int
    charges: List[SolventCharge]

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
