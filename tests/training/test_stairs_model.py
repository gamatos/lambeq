
import numpy

from lambeq import StairsModel


def test_weights():
    test_sentences = [['Alice', 'likes', 'the', 'city', 'park'],
                      ['Bob', 'likes', 'Alice']]

    test_model = StairsModel.from_tokenised_sentences(test_sentences)

    test_model.initialise_weights()

    # Test whether the correct number of weights have been generated
    assert test_model.weights.shape[0] == len(
        {s for ts in test_sentences for s in ts})

    # Test whether weight padding is being done correctly
    assert(numpy.isclose(test_model.padded_weights[1:, :-1],
                         test_model.weights)).all()

    new_sentences = [['Alice', 'likes', 'Bob'],
                     ['Bob', 'likes', 'the', 'park']]
    i = test_model._indices_from_diagrams(new_sentences)

    d = test_model.word_dict

    # Test whether weight indices are being correctly extracted from sentences
    assert len(i) == len({s for ts in new_sentences for s in ts})
    assert (i == sorted({d['Alice'], d['likes'], d['the'],
                         d['park'], d['Bob']})).all()

    bi = test_model._batched_weight_indices(new_sentences)
    bw = test_model._indices_to_angles(bi)

    max_words = max(len(s) for s in test_sentences)

    # Test whether batched weights have the correct nr of sentences,
    # words and angles
    assert bw.shape == (len(new_sentences), max_words, 5)

    # Test whether the same words have the same weights
    assert numpy.isclose(bw[0, 2, :], bw[1, 0, :]).all()
    assert numpy.isclose(bw[0, 1,:], bw[1, 1,:]).all()

    padding_1 = max_words - len(new_sentences[0])
    padding_2 = max_words - len(new_sentences[1])

    # Test whether weight padding is being performed correctly
    assert numpy.isclose(bw[0, -padding_1:, :],
                         numpy.zeros((padding_1, 5))).all()
    assert numpy.isclose(bw[1, -padding_2:, :],
                         numpy.zeros((padding_2, 5))).all()

    assert numpy.isclose(bw[0, :-padding_1, -1],
                         numpy.ones(len(new_sentences[0]))).all()
    assert numpy.isclose(bw[1, :-padding_2:, -1],
                         numpy.ones(len(new_sentences[1]))).all()


def test_mask():
    test_sentences = [['Alice', 'likes', 'the', 'city', 'park'],
                      ['Bob', 'likes', 'Alice']]

    test_model = StairsModel.from_tokenised_sentences(test_sentences)

    test_model.initialise_weights()

    new_sentences = [['Alice', 'likes', 'Bob'],
                     ['Bob', 'likes', 'the', 'park']]

    mask = test_model._relevant_parameter_mask(new_sentences)

    alice_index = test_model.word_dict["Alice"]
    bob_index = test_model.word_dict["Bob"]

    i = test_model._indices_from_diagrams(new_sentences)
    i = numpy.delete(i, alice_index)

    # Test whether mask is correctly identifying weights relevant to
    # the words in the sentences
    assert numpy.allclose(mask[alice_index], numpy.array([0, 0, 0, 1]))
    assert numpy.allclose(mask[bob_index], numpy.array([0, 0, 0, 0]))
    assert numpy.all(mask[i] == 0)


def test_IQP_circuit():
    angles = numpy.array([[0.687, 0.254, 0.555, 0.182, 1.     ],
                          [0.065, 0.54 , 0.129, 0.182, 1.     ],
                          [0.914, 0.287, 0.87 , 0.182, 1.     ],
                          [0.   , 0.   , 0.   , 0.   , 0.     ],])

    result = numpy.square(numpy.abs(
        numpy.array([0.39741468+0.24029578j,  0.86303266-0.19873134j])))

    from lambeq.training.stairs_model import _IQP_ansatz_evaluator

    f = _IQP_ansatz_evaluator()

    assert numpy.isclose(f(angles), result).all()

    # Test whether initial stairs angle is being properly ignored
    angles[0, 3] = 100000
    assert numpy.isclose(f(angles), result).all()

    """
    Value of result obtained by manually coding the following
    circuit in qiskit:
        ┌───────────┐┌───────────┐ ┌──────────┐┌───┐             ┌───┐»
   q_0: ┤ Rx(1.374) ├┤ Rz(0.508) ├─┤ Rx(1.11) ├┤ H ├──────■──────┤ H ├»
        └┬──────────┤└┬──────────┤┌┴──────────┤├───┤┌─────┴─────┐└───┘»
   q_1: ─┤ Rx(0.13) ├─┤ Rz(1.08) ├┤ Rx(0.258) ├┤ H ├┤ Rz(0.364) ├─|0>─»
         └──────────┘ └──────────┘└───────────┘└───┘└───────────┘     »
   c: 2/══════════════════════════════════════════════════════════════»
                                                                      »
   «
   «q_0: ─────────────────────────────────────────────────■───────────
   «     ┌───────────┐┌───────────┐┌──────────┐┌───┐┌─────┴─────┐
   «q_1: ┤ Rx(1.828) ├┤ Rz(0.574) ├┤ Rx(1.74) ├┤ H ├┤ Rz(0.364) ├─|0>─
   «     └───────────┘└───────────┘└──────────┘└───┘└───────────┘
   «c: 2/═════════════════════════════════════════════════════════════
   «
    """


def test_sentence_evaluation():
    from lambeq.training.stairs_model import _IQP_ansatz_evaluator
    from jax import vmap

    f = _IQP_ansatz_evaluator()
    vf = vmap(f)

    test_sentences = [['Alice', 'likes', 'the', 'city', 'park'],
                      ['Bob', 'likes', 'Alice']]

    test_model = StairsModel.from_tokenised_sentences(test_sentences)

    test_model.initialise_weights()

    d = test_model.word_dict

    new_sentence = [['Alice', 'likes', 'Bob']]
    new_sentence_indices = numpy.array([d['Alice']+1, d['likes']+1,
                                        d['Bob']+1])
    w = test_model.padded_weights[new_sentence_indices]

    bi = test_model._batched_weight_indices(new_sentence)
    bw = test_model._indices_to_angles(bi)

    # Tests whether weight batching gives the same result as
    # manually constructed unbatched evaluation
    assert numpy.isclose(vf(bw)[0], f(w)).all()

    # Test whether model evaluation gives the same result as evaluation
    # directly on circuit
    assert numpy.isclose(vf(bw), test_model(new_sentence)).all()
