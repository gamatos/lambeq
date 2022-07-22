# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
StairsModel
==========
A specialized lambeq model that evaluates diagrams having a staircase
structure.

As of right now, it bypasses DisCoPy, assuming that the underlying
diagrams have a staircase structure (i.e. as if they had been created 
with the `lambeq.stairs_reader` function). This is leveraged so that
JAX can be efficiently used, as only one function needs to be compiled
for the entire model, as opposed to one per circuit.

The `IQPAnsatz` is currently hard-coded into the model.

There is a minor difference between the diagrams created through
`lambeq.stairs_reader` and the structure assumed in this model,
as here the parameters associated to each STAIR box are allowed
to vary independently.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import jax
import numpy


def _IQP_ansatz_evaluator():
    """
    Returns a function that takes a set of angles and evaluates
    the `IQPAnsatz` with one qubit and one layer
    
    The angles should be supplied as a `jax.numpy.ndarray`.
    Padding is represented as a row having all zeros.
    """
    import jax.numpy as jnp

    # Hard coding of the IQPAnsatz
    # TODO: Make this more flexible to allow for other ansatze
    # This could be achieved by introducing an argument to this function
    # that specifies the ansatz

    initial_state = jnp.array([1, 0, 0, 0])

    H = 1/jnp.sqrt(2)*jnp.array([[1, 1], [1, -1]])
    HH = jnp.kron(H, H)
    Id_local = jnp.eye(2)
    proj = jnp.kron(jnp.array([[1., 0.], [0., 0.]]), Id_local)
    Id = jnp.eye(4)

    def _crz(t):
        return jnp.diag(jnp.array([1,  jnp.exp(-1j * t),
                                   1,  jnp.exp(1j * t)]))

    def _rz(t):
        return jnp.diag(jnp.array([jnp.exp(-1j*t), jnp.exp(1j*t)]))

    def _rx(t):
        return jnp.array([[jnp.cos(t), -1j*jnp.sin(t)],
                          [-1j*jnp.sin(t), jnp.cos(t)]])

    def _qubit_rotations(x):
        rot = _rx(x[2]) @ _rz(x[1]) @ _rx(x[0])
        return rot

    def apply_stair_box(carry, x):
        # Flag indicating whether current word is padding
        pad = x[4]

        rot = jnp.kron(_qubit_rotations(x[:3]), Id_local)
        res = rot @ carry
        res = (pad * HH + (1-pad) * Id) @ res
        res = _crz(x[3]) @ res
        res = (pad * proj + (1-pad) * Id) @ res
        res *= 1/jnp.linalg.norm(res)
        return res, None

    def evaluate_circuit(angles : jax.numpy.ndarray):
        rot0 = jnp.kron(Id_local, _qubit_rotations(angles[0, :3]))
        x0 = rot0 @ initial_state
        res = jax.lax.scan(apply_stair_box, x0, angles[1:,:])[0]
        return jnp.abs(jnp.square(res.reshape(2, 2)[0, :]))

        # The above jax.lax.scan call is equivalent to:
        # x = x0
        # print(x0)
        # for a in angles[1:,:]:
        #     x, _ = apply_stair_box(x, a)
        #     print(x)
        # return x

    return evaluate_circuit


# Batch and compile function that evaluates IQP ansatz
IQP_ansatz_evaluator = jax.jit(jax.vmap(_IQP_ansatz_evaluator()))


class StairsModel():
    """A specialized lambeq model for training of a stair diagram
    based model."""

    # TODO: Incorporate different ansatze
    # Currently the IQPAnsatz is hard-coded in this class

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __init__(self, word_list : Iterable[str], max_sentence_length : int,
                 **kwargs) -> None:
        """Initialise a StairsModel.

        Parameters
        ----------
        word_list : Iterable[str]
            List of words that make up the model
        word_list : int
            The maximum admissible sentence length
            Used for padding purposed

        """
        self.word_dict = {w: i for i, w in enumerate(set(word_list))}
        self.max_sentence_length = max_sentence_length
        self.padded_weights = numpy.array([[0]])

    def pad_weights(self, weights : jax.numpy.ndarray):
        """
        Takes a set of weights represented by an array and returns a
        padded version.

        This is done by adding rows containing zeros so the nr. of rows
        matches the maximum sentence length. A column signaling whether
        the corresponding row is a padding row is also included.

        This structure is needed for batched evaluation.
        """

        nr_of_words = len(self.word_dict)

        # Extra column/row to deal with padding
        padded_weights = numpy.c_[weights, numpy.ones(nr_of_words)]
        padded_weights = numpy.vstack([numpy.zeros(5), padded_weights])

        return padded_weights

    def initialise_weights(self,
                           generator : Callable = numpy.random.rand) -> None:
        """
        Initialise the weights of the model.

        Parameters
        ----------
        generator : Callable, default = numpy.random.rand
            Callable that generates model's weigths

        """
        nr_of_words = len(self.word_dict)
        weights = generator(nr_of_words, 4)

        padded_weights = self.pad_weights(weights)

        self.padded_weights = padded_weights

    def _batched_weight_indices(self,
                                tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns the angles needed
        to evaluate the ansatz as a single array to be supplied to
        the batched evaluation function.
        """

        indices = [[self.word_dict[w] + 1 for w in ts]
                   for ts in tokenised_sentences]

        # Add indices corresponding to padding
        for i in indices:
            i += [0] * (self.max_sentence_length - len(i))

        indices = numpy.array(indices)

        return indices

    def _indices_from_diagrams(self,
                               tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns the indices to the
        array of weights corresponding to the words featuring in the
        sentence.
        """

        indices = sorted({self.word_dict[w]
                         for ts in tokenised_sentences for w in ts})
        indices = numpy.array(indices)

        return indices

    def _relevant_parameter_mask(self,
                                 tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns a mask that
        indicating what weights are relevant
        """

        # Create and apply mask
        i = self._indices_from_diagrams(tokenised_sentences)
        mask = numpy.ones(self.weights.shape, dtype=int)
        mask[i, :] = 0

        # Mask angles that would be used for first STAIR box
        remaining_words = [w for t in tokenised_sentences for w in t[1:]]
        first_words = [t[0] for t in tokenised_sentences
                       if t[0] not in remaining_words]
        if len(first_words) > 0:
            i_stair = self._indices_from_diagrams([first_words])
            mask[i_stair, 3] = 1

        return mask

    def _indices_to_angles(self, indices: jax.numpy.ndarray):
        """
        Converts a set of word indices into the corresponding angles
        """
        batched_angles = self.padded_weights[indices]
        return batched_angles

    def get_diagram_output(
                self,
                tokenised_sentences: Iterable[Iterable[str]]) -> numpy.ndarray:
        """Return the exact prediction for each sentence.

        Parameters
        ----------
        sentence : Iterable of Iterable of str
            The list of sentences to be evaluated.


        Returns
        -------
        jax.numpy.ndarray
            Resulting array containing results.

        """
        indices = self._batched_weight_indices(tokenised_sentences)

        batched_angles = self._indices_to_angles(indices)

        return IQP_ansatz_evaluator(batched_angles)

    def forward(self, x: Iterable[Iterable[str]]) -> numpy.ndarray:
        """Perform forward pass of model.

        Parameters
        ----------
        x : Iterable of strings
            The sentences to be evaluated.

        Returns
        -------
        jax.numpy.ndarray
            Array containing model's prediction.

        """
        return self.get_diagram_output(x)

    @classmethod
    def from_tokenised_sentences(
                cls,
                tokenised_sentences: Iterable[Iterable[str]]) -> numpy.ndarray:
        """
        Create model from a set of tokenised sentences
        """

        words = [w for s in tokenised_sentences for w in s]
        return cls(words, max(map(len, tokenised_sentences)))

    @property
    def weights(self):
        weights = self.padded_weights[1:, :-1]
        return weights

    @weights.setter
    def weights(self, new):
        self.padded_weights[1:, :-1] = new

    @property
    def symbols(self):
        return list(self.word_dict.keys())

    def from_checkpoint():
        raise NotImplementedError()
