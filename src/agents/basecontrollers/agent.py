from abc import abstractmethod

from abc import abstractmethod
import dataclasses
import inspect
import os
from abc import abstractmethod

import pickle
import flax
import jax

class Obj:

  def freeze(self):
    object.__setattr__(self, "__frozen__", True)

  def unfreeze(self):
    object.__setattr__(self, "__frozen__", False)

  def is_frozen(self):
    return not hasattr(self, "__frozen__") or getattr(self, "__frozen__")

  def __new__(cls, *args, **kwargs):
    """A true bastardization of __new__..."""

    def __setattr__(self, name, value):
      if self.is_frozen():
        raise dataclasses.FrozenInstanceError
      object.__setattr__(self, name, value)

    def replace(self, **updates):
      obj = dataclasses.replace(self, **updates)
      obj.freeze()
      return obj

    cls.__setattr__ = __setattr__
    cls.replace = replace

    obj = object.__new__(cls)
    obj.unfreeze()

    return obj

  @classmethod
  def __init_subclass__(cls, *args, **kwargs):
    flax.struct.dataclass(cls)

  @classmethod
  def create(cls, *args, **kwargs):
    # NOTE: Oh boy, this is so janky
    obj = cls(*args, **kwargs)
    obj.setup()
    obj.freeze()

    return obj

  @classmethod
  def unflatten(cls, treedef, leaves):
    """Expost a default unflatten method"""
    return jax.tree_util.tree_unflatten(treedef, leaves)

  def setup(self):
    """Used in place of __init__"""

  def flatten(self):
    """Expose a default flatten method"""
    return jax.tree_util.tree_flatten(self)[0]


class Env(Obj):

  @abstractmethod
  def init(self):
    """Return an initialized state"""

  @abstractmethod
  def __call__(self, state, action, *args, **kwargs):
    """Return an updated state"""


class AgentState(Obj):
  time: float = float("inf")
  steps: int = 0


class Agent(Obj):

  @abstractmethod
  def __call__(self, state, obs, *args, **kwargs):
    """Return an updated state"""

  def init(self):
    return AgentState()

class DAC:

    @abstractmethod
    def update(self, state, u, A, B):
        """Return an updated state"""

    @abstractmethod
    def get_action(self, state):
        """Get Action given """


class DRC:

    @abstractmethod
    def update(self, G, x, u):
        """Return an updated state"""

    @abstractmethod
    def get_action(self):
        """Get Action given """
