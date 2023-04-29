# -*- coding: utf-8 -*-
import os
import trax
import jax
import jaxlib
import numpy as np
import pickle
import random
import textwrap


from trax import shapes
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.supervised import training
from jax.random import PRNGKey


class next_symbol_mocker:
    """Class that represents a mock of the funcion next_symbol.     


    Attributes:    
        path_test_files (str): path of directory that contains .pkl files.    

    Methods:    
        read_path_test_files(): Reads the files in .pkl format with 
            the actual input/output mapping

        mocked_fn(NMTAttn=None
            , input_tokens=None
            , cur_output_tokens=None
            , temperature=0.0): Returns the input/output mapping.
    """

    def __init__(self, path_test_files):
        self.path_test_files = path_test_files
        self.dict_output = self.read_path_test_files()

    def read_path_test_files(self):
        """Reads files in .pkl format.
                
        Returns:
            dict: Dictionary that maps the input and output directly.
        """

        with open(self.path_test_files, "rb") as f:
            dict_raw = pickle.load(f)

        return dict_raw

    def mocked_fn(self, cur_output_tokens, model=None):
        return self.dict_output[tuple(cur_output_tokens)]


def test_DotProductAttention(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "test dummy tensors",
            "input": {
                "query": jnp.array([[1, 0, 0], [0, 1, 0]]),
                "key": jnp.array([[1, 2, 3], [4, 5, 6]]),
                "value": jnp.array([[0, 1, 0], [1, 0, 1]]),
                "mask": jnp.array([[0, 0], [-1e9, 0]]),
            },
            "expected": jnp.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
            "error": "Expected output for DotProductAttention does not match.",
        },
        {
            "name": "test dummy tensors",
            "input": {
                "query": jnp.array([[1, 2, 3], [4, 5, 6]]),
                "key": jnp.array([[1, 2, 3], [4, 5, 6]]),
                "value": jnp.array([[0, 1, 1], [0, 0, 1]]),
                "mask": jnp.array([[0, 0], [-1e9, 0]]),
            },
            "expected": jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 1.0]]),
            "error": "Expected output for DotProductAttention does not match.",
        },
        {
            "name": "test dummy tensors",
            "input": {
                "query": jnp.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]]),
                "key": jnp.array([[9, 8, 7, 6], [6, 5, 4, 3], [3, 2, 1, 0]]),
                "value": jnp.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1]]),
                "mask": jnp.array([[0, 0, 0], [-1e9, 0, 0], [-1e9, -1e9, 0]]),
            },
            "expected": jnp.array(
                [[1, 1, 1.0, 1], [0, 0, 1, 0], [0.04742587, 0, 0.95257413, 0]]
            ),
            "error": "Expected output for DotProductAttention does not match.",
        },
        {
            "name": "test dummy tensors",
            "input": {
                "query": jnp.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]]),
                "key": jnp.array([[9, 8, 7, 6], [6, 5, 4, 3], [3, 2, 1, 0]]),
                "value": jnp.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1]]),
                "mask": jnp.array([[-1e9, -1e9, -1e9], [0, -1e9, -1e9], [0, 0, -1e9]]),
            },
            "expected": jnp.array(
                [
                    [0.04731418, 0.00235563, 0.9503307, 0.00235563],
                    [0.8175744, 0.1824255, 0.0, 0.1824255],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            ),
            "error": "Expected output for DotProductAttention does not match.",
        },
    ]

    for test_case in test_cases:
        name = test_case.get("name")

        input_dict = test_case.get("input")
        expected = test_case.get("expected")
        output = target(**input_dict)

        try:
            assert np.isclose(output, expected).all()
            successful_cases += 1
        except:
            print(test_case.get("error"))
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": output,
                }
            )
            print(
                f"Wrong output from DotProductAttention. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        if name == "test dummy tensors":
            try:
                assert isinstance(output, jax.Array)
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": jax.Array,
                        "got": type(output),
                    }
                )
                print(
                    f"Output from DotProductAttention is of type {type(output)}. Expected type: jax.Array"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_compute_attention_heads_closure(target):
    n_heads = 2
    d_head = 3
    successful_cases = 0
    failed_cases = []

    q = jnp.array([[1, 0, 0], [0, 1, 0]])

    test_cases = [
        {
            "name": "test dummy tensors",
            "input": {
                "x": jnp.array(
                    [jnp.concatenate([q, q], axis=-1), jnp.concatenate([q, q], axis=-1)]
                )
            },
            "expected": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                ]
            ),
            "error_message": [
                "Expected shape does not match",
                "Expected output does not match.",
            ],
        },
        {
            "name": "test dummy tensors",
            "input": {
                "x": jnp.array(
                    [
                        jnp.concatenate([q, q], axis=-1),
                        jnp.concatenate([q, q], axis=-1),
                        jnp.concatenate([q, q], axis=-1),
                    ]
                )
            },
            "expected": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                ]
            ),
            "error_message": [
                "Expected shape does not match",
                "Expected output does not match.",
            ],
        },
    ]

    for test_case in test_cases:
        name = test_case.get("name")

        input_dict = test_case.get("input")
        expected = test_case.get("expected")
        output = target(n_heads=n_heads, d_head=d_head)(**input_dict)

        try:
            assert output.shape == expected.shape
            successful_cases += 1
        except:
            print(test_case.get("error")[0])
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": output.shape,
                }
            )

        try:
            assert jnp.isclose(output, expected).all()
            successful_cases += 1
        except:
            print(test_case.get("error")[1])
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": output,
                }
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_dot_product_self_attention(target):
    n_heads = 2
    d_head = 3
    successful_cases = 0
    failed_cases = []

    q = jnp.array([[1, 0, 0], [0, 1, 0]])
    k = jnp.array([[1, 2, 3], [4, 5, 6]])
    v = jnp.array([[0, 1, 0], [1, 0, 1]])
    m = jnp.array([[0, 0], [-1e9, 0]])

    test_cases = [
        {
            "name": "test dummy tensors",
            "input": {"q": q[None, :], "k": k[None, :], "v": v[None, :],},
            "expected": jnp.array(
                [[[0.0, 1.0, 0.0], [0.8496746, 0.15032543, 0.8496746]]]
            ),
            "error_message": [
                "Expected shape does not match",
                "Expected output does not match.",
            ],
        },
        {
            "name": "test dummy tensors",
            "input": {
                "q": jnp.array([jnp.concatenate([q, q], axis=-1)]),
                "k": jnp.array([jnp.concatenate([k, k], axis=-1)]),
                "v": jnp.array([jnp.concatenate([v, v], axis=-1)]),
            },
            "expected": jnp.array(
                [
                    [
                        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        [
                            0.9205239,
                            0.07947586,
                            0.9205239,
                            0.9205239,
                            0.07947586,
                            0.9205239,
                        ],
                    ]
                ]
            ),
            "error_message": [
                "Expected shape does not match",
                "Expected output does not match.",
            ],
        },
    ]

    for test_case in test_cases:
        name = test_case.get("name")

        input_dict = test_case.get("input")
        expected = test_case.get("expected")
        output = target(**input_dict)

        try:
            assert output.shape == expected.shape
            successful_cases += 1
        except:
            print(test_case.get("error")[0])
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": output.shape,
                }
            )

        try:
            assert jnp.isclose(output, expected).all()
            successful_cases += 1
        except:
            print(test_case.get("error")[1])
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": output,
                }
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_compute_attention_output_closure(target):
    n_heads = 2
    d_head = 3
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "test dummy tensors 1",
            "input": {
                "x": jnp.array(
                    [
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                    ]                
                )
                , "n_heads": 2
                , "d_head": 3
            },
            "expected": jnp.array(
                [
                    [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]],
                    [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]],
                    [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]],
                ]
            ),
            "error_message": [
                "Expected shape does not match",
                "Expected output does not match.",
            ],
        },
        {
            "name": "test dummy tensors 2",
            "input": {
                "x": jnp.array(
                    [
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [0, 1, 0]],
                    ]
                )
                , "n_heads": 3
                , "d_head": 2
            },
            "expected": jnp.array(
                [[[1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]],
                 [[1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]],                 
                 [[1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]]]
            ),
            "error_message": [
                "Expected shape does not match",
                "Expected output does not match.",
            ],
        },
    ]

    for test_case in test_cases:
        name = test_case.get("name")

        input_dict = test_case["input"]
        expected = test_case.get("expected")
        output = target(n_heads=input_dict['n_heads'], d_head=input_dict['d_head'])(input_dict['x'])

        try:
            assert output.shape == expected.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": output.shape,
                }
            )

        try:
            assert jnp.isclose(output, expected).all()
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": output,
                }
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_CausalAttention(target):
    
    successful_cases = 0
    failed_cases = []

    test_cases = [{'name': 'default_causal_attention'
                  , 'input': {'d_feature':512, 'n_heads':8}
                  , 'expected': {'str_rep': "Serial[\n  Branch_out3[\n    [Dense_512, AttnHeads]\n    [Dense_512, AttnHeads]\n    [Dense_512, AttnHeads]\n  ]\n  DotProductAttn_in3\n  AttnOutput\n  Dense_512\n]"
                                  , 'sublayers': (trax.layers.Serial,
                                    [
                                        trax.layers.PureLayer,
                                        trax.layers.PureLayer,
                                        trax.layers.Serial,
                                        trax.layers.Dense,
                                    ]
                                ),
                                'type': trax.layers.combinators.Serial
                                }
                  }
                  ,{'name': 'small_causal_attention'
                      , 'input': {'d_feature':16, 'n_heads':4}
                      , 'expected': {'str_rep': "Serial[\n  Branch_out3[\n    [Dense_16, AttnHeads]\n    [Dense_16, AttnHeads]\n    [Dense_16, AttnHeads]\n  ]\n  DotProductAttn_in3\n  AttnOutput\n  Dense_16\n]"
                                      , 'sublayers': (trax.layers.Serial,
                                        [
                                            trax.layers.PureLayer,
                                            trax.layers.PureLayer,
                                            trax.layers.Serial,
                                            trax.layers.Dense,
                                        ]
                                    ),
                                    'type': trax.layers.combinators.Serial
                                    }
                  }
                 ]
    
    for test_case in test_cases:
        output = target(**test_case['input'])
    
        try:
            assert str(output) == test_case['expected']['str_rep']
            successful_cases += 1
        except: 
            failed_cases.append(
                {
                    "name": "model_check",
                    "expected": test_case['expected']['str_rep'],
                    "got": str(output),
                }
            )
            print(
                f"Causal Attention layer is correctly defined {failed_cases[-1].get('expected')}"
            )

        test_func = lambda x: (type(x), sorted(map(type, x.sublayers), key=str))

        try:
            assert isinstance(output, test_case['expected']['type'])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "type_check",
                    "expected": test_case['expected']['type'],
                    "got": type(output),
                }
            )
            print(
                f"Causal Attention layer is not an object of {failed_cases[-1].get('expected')}"
            )

        try:
            assert test_func(output) == test_case['expected']['sublayers']
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayers_type_check",
                    "expected": expected,
                    "got": test_func(output),
                }
            )
            print(
                f"Wrong model was defined. Expected{failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert len(output.sublayers) == len(test_case['expected']['sublayers'][1])
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "n_sublayers_check", "expected": len(test_case['expected']['sublayers'][1]), "got": len(output.sublayers),}
            )
            print(
                f"The number of sublayers does not match. Expected: {len(test_case['expected']['sublayers'][1])}. Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_DecoderBlock(target):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [{'name': 'default_decoder'
                   ,'input': {'d_model': 512, 'd_ff': 2048, 'n_heads': 8, 'dropout': 0.1, 'mode': 'train', 'ff_activation': tl.Relu}
                   , 'expected': {'expected_str': '[Serial[\n  Branch_out2[\n    None\n    Serial[\n      LayerNorm\n      Serial[\n        Branch_out3[\n          [Dense_512, AttnHeads]\n          [Dense_512, AttnHeads]\n          [Dense_512, AttnHeads]\n        ]\n        DotProductAttn_in3\n        AttnOutput\n        Dense_512\n      ]\n      Dropout\n    ]\n  ]\n  Add_in2\n], Serial[\n  Branch_out2[\n    None\n    Serial[\n      LayerNorm\n      Dense_2048\n      Serial[\n        Relu\n      ]\n      Dropout\n      Dense_512\n      Dropout\n    ]\n  ]\n  Add_in2\n]]'
                                  , 'expected_types': [
                                                        (
                                                            trax.layers.combinators.Serial,
                                                            [trax.layers.base.PureLayer, trax.layers.combinators.Serial],
                                                        ),
                                                        (
                                                            trax.layers.combinators.Serial,
                                                            [trax.layers.base.PureLayer, trax.layers.combinators.Serial],
                                                        ),
                                                    ]
                                }
                  }
                  , {'name': 'default_decoder'
                    ,'input': {'d_model': 16, 'd_ff': 32, 'n_heads': 4, 'dropout': 0.05, 'mode': 'train', 'ff_activation': tl.Relu}
                    , 'expected': {'expected_str': '[Serial[\n  Branch_out2[\n    None\n    Serial[\n      LayerNorm\n      Serial[\n        Branch_out3[\n          [Dense_16, AttnHeads]\n          [Dense_16, AttnHeads]\n          [Dense_16, AttnHeads]\n        ]\n        DotProductAttn_in3\n        AttnOutput\n        Dense_16\n      ]\n      Dropout\n    ]\n  ]\n  Add_in2\n], Serial[\n  Branch_out2[\n    None\n    Serial[\n      LayerNorm\n      Dense_32\n      Serial[\n        Relu\n      ]\n      Dropout\n      Dense_16\n      Dropout\n    ]\n  ]\n  Add_in2\n]]'
                                  , 'expected_types': [
                                                        (
                                                            trax.layers.combinators.Serial,
                                                            [trax.layers.base.PureLayer, trax.layers.combinators.Serial],
                                                        ),
                                                        (
                                                            trax.layers.combinators.Serial,
                                                            [trax.layers.base.PureLayer, trax.layers.combinators.Serial],
                                                        ),
                                                    ]
                                }
                  }
                 ]


    for test_case in test_cases:
        output = target(**test_case['input'])
        
        test_func = lambda x: (type(x), sorted(map(type, x.sublayers), key=str))

        try:
            assert str(output).replace(" ", "") == test_case['expected']['expected_str'].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "len_check", "expected": test_case['expected']['expected_str'], "got": str(output),}
            )
            print(
                "Wrong model. \nProposed:\n%s" % failed_cases[-1].get("got"),
                "\nExpected:\n%s" % failed_cases[-1].get("expected"),
            )

        try:
            assert len(output) == len(test_case['expected']['expected_types'])
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "len_check", "expected": len(test_case['expected']['expected_types']), "got": len(output),}
            )
            print(
                f"Decoder Block list does not have the correct number of elements. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(output[0], trax.layers.combinators.Serial)
            assert isinstance(output[1], trax.layers.combinators.Serial)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "type_check",
                    "expected": (
                        trax.layers.combinators.Serial,
                        trax.layers.combinators.Serial,
                    ),
                    "got": (type(output[0]), type(output[1])),
                }
            )
            print(
                f"Decoder Block list elements do not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert test_func(output[0]) == test_case['expected']['expected_types'][0]
            assert test_func(output[1]) == test_case['expected']['expected_types'][1]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayers_type_check",
                    "expected": test_case['expected']['expected_types'],
                    "got": [test_func(output[0]), test_func(output[1])],
                }
            )
            print(
                f"Decoder Block sublayers do not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert len(output[0].sublayers) == len(test_case['expected']['expected_types'][0][1])
            assert len(output[1].sublayers) == len(test_case['expected']['expected_types'][1][1])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "n_sublayers_check",
                    "expected": [len(test_case['expected']['expected_types'][0][1]), len(test_case['expected']['expected_types'][1][1])],
                    "got": [len(output[0].sublayers), len(output[1].sublayers)],
                }
            )
            print(
                f"The number of sublayers does not match. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_TransformerLM(target):

    test_cases = [{'name': 'default_decoder'
                   ,'input': { 'n_layers': 1, 'vocab_size': 33300, 'd_model': 512, 'd_ff': 2048, 'n_heads': 8, 'dropout': 0.1, 'max_len': 4096, 'ff_activation': tl.Relu}
                   , 'expected': { 'expected_str': 'Serial[\n  Serial[\n    ShiftRight(1)\n  ]\n  Embedding_33300_512\n  Dropout\n  PositionalEncoding\n  Serial[\n    Branch_out2[\n      None\n      Serial[\n        LayerNorm\n        Serial[\n          Branch_out3[\n            [Dense_512, AttnHeads]\n            [Dense_512, AttnHeads]\n            [Dense_512, AttnHeads]\n          ]\n          DotProductAttn_in3\n          AttnOutput\n          Dense_512\n        ]\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  Serial[\n    Branch_out2[\n      None\n      Serial[\n        LayerNorm\n        Dense_2048\n        Serial[\n          Relu\n        ]\n        Dropout\n        Dense_512\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  LayerNorm\n  Dense_33300\n  LogSoftmax\n]'
                                  , 'expected_types': (
                                                        trax.layers.combinators.Serial,
                                                        [
                                                            trax.layers.attention.PositionalEncoding,
                                                            trax.layers.base.PureLayer,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.core.Dense,
                                                            trax.layers.core.Dropout,
                                                            trax.layers.core.Embedding,
                                                            trax.layers.normalization.LayerNorm,
                                                        ],
                                                    )
                                 }
                  }
                  ,{'name': 'default_decoder'
                   ,'input': { 'n_layers': 1, 'vocab_size': 100, 'd_model': 16, 'd_ff': 32, 'n_heads': 4, 'dropout': 0.05, 'max_len': 256, 'ff_activation': tl.Relu}
                   , 'expected': { 'expected_str': 'Serial[\n  Serial[\n    ShiftRight(1)\n  ]\n  Embedding_100_16\n  Dropout\n  PositionalEncoding\n  Serial[\n    Branch_out2[\n      None\n      Serial[\n        LayerNorm\n        Serial[\n          Branch_out3[\n            [Dense_16, AttnHeads]\n            [Dense_16, AttnHeads]\n            [Dense_16, AttnHeads]\n          ]\n          DotProductAttn_in3\n          AttnOutput\n          Dense_16\n        ]\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  Serial[\n    Branch_out2[\n      None\n      Serial[\n        LayerNorm\n        Dense_32\n        Serial[\n          Relu\n        ]\n        Dropout\n        Dense_16\n        Dropout\n      ]\n    ]\n    Add_in2\n  ]\n  LayerNorm\n  Dense_100\n  LogSoftmax\n]'
                                  , 'expected_types': (
                                                        trax.layers.combinators.Serial,
                                                        [
                                                            trax.layers.attention.PositionalEncoding,
                                                            trax.layers.base.PureLayer,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.combinators.Serial,
                                                            trax.layers.core.Dense,
                                                            trax.layers.core.Dropout,
                                                            trax.layers.core.Embedding,
                                                            trax.layers.normalization.LayerNorm,
                                                        ],
                                                    )
                                 }
                  }
        
                ]
    
    successful_cases = 0
    failed_cases = []
    
    for test_case in test_cases:
        output = target(**test_case['input'])
    
        test_func = lambda x: (type(x), sorted(map(type, x.sublayers), key=str))

        try:
            assert str(output).replace(" ", "") == test_case['expected']['expected_str'].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "str_check", "expected": test_case['expected']['expected_str'], "got": str(output),}
            )
            print(
                f"Wrong model. \nProposed:\n {failed_cases[-1].get('got')}. \nExpected:\n {failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(output, trax.layers.combinators.Serial)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "type_check",
                    "expected": trax.layers.combinators.Serial,
                    "got": type(output),
                }
            )
            print(
                f"TransformerLM does not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:                        
            assert test_func(output)[0] == test_case['expected']['expected_types'][0]
            for i in range(len(test_case['expected']['expected_types'][1])):                
                assert str(test_func(output)[1][i]) == str(test_case['expected']['expected_types'][1][i])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "sublayers_type_check",
                    "expected": test_case['expected']['expected_types'],
                    "got": test_func(output),
                }
            )
            print(
                f"TransformerLM sublayers do not have the correct type. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

        try:
            assert len(output.sublayers) == len(test_case['expected']['expected_types'][1])
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": "n_sublayers_check", "expected": len(test_case['expected']['expected_types'][1]), "got": len(output.sublayers),}
            )
            print(
                f"The number of sublayers does not match. Expected: {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_training_loop(target, TransformerLM):

    successful_cases = 0
    failed_cases = []

    def my_gen():
        random.seed(54321)
        while True:
            lng1 = random.randrange(1000, 1300)
            sample1 = np.random.randint(2, 30000, lng1)
            pos1 = random.randrange(int(3 * lng1 / 4), lng1)
            sample1[pos1] = 1
            sample1[lng1 - 1] = 1
            sample3 = np.concatenate(
                (np.zeros_like(sample1[:pos1]), np.ones_like(sample1[pos1:])), axis=0
            )
            sample1 = sample1[None, :]
            sample3 = sample3[None, :]
            sample2 = sample1.copy()
            yield sample1, sample2, sample3

            
    if os.path.exists("~/model/model.pkl.gz"):
        os.remove("~/model/model.pkl.gz")
    
    output_loop = target(TransformerLM, my_gen(), my_gen())

    try:
        strlabel = str(output_loop._tasks[0]._loss_layer)
        assert strlabel == "CrossEntropyLoss_in3"
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "loss_layer_check",
                "expected": "CrossEntropyLoss_in3",
                "got": strlabel,
            }
        )
        print(
            f"Wrong loss functions. CrossEntropyLoss_in3 was expected. Got {failed_cases[-1].get('got')}."
        )

    # Test the optimizer parameter
    try:
        assert isinstance(output_loop._tasks[0].optimizer, trax.optimizers.adam.Adam)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_check",
                "expected": trax.optimizers.adam.Adam,
                "got": type(output_loop._tasks[0].optimizer),
            }
        )
        print(
            f"Wrong optimizer. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )
        
    opt_params_dict = {'weight_decay_rate': jnp.array(1.e-5),
                         'b1': jnp.array(0.9),
                         'b2': jnp.array(0.999),
                         'eps': jnp.array(1.e-5),
                         'learning_rate': jnp.array(0.01)}
    
    try: 
        assert output_loop._tasks[0].optimizer.opt_params == opt_params_dict
        successful_cases += 1
    except:
        failed_cases.append({"name": "optimizer_parameters",
                            "expected": opt_params_dict,
                            "got": output_loop._tasks[0].optimizer.opt_params,})
        
        print(f"Optimizer has the wrong parameters.\n\tExpected {opt_params_dict}.\n\tGot {output_loop._tasks[0].optimizer.opt_params}.")
            
    # Test the schedule parameter
    try:
        assert isinstance(
            output_loop._tasks[0]._lr_schedule,
            trax.supervised.lr_schedules._BodyAndTail,
        )
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "lr_schedule_check",
                "expected": trax.supervised.lr_schedules._BodyAndTail,
                "got": type(output_loop._tasks[0]._lr_schedule),
            }
        )
        print(
            f"Wrong learning rate schedule type. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )
    
    try: 
        assert output_loop._tasks[0]._lr_schedule._body_value == 0.01
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "lr_check",
                "expected": 0.01,
                "got": output_loop._tasks[0]._lr_schedule._body_value,
            }
        )
        print(f'Wrong learning rate value.\n\tExpected 0.01.\n\tGot {output_loop._tasks[0]._lr_schedule._body_value}.')


    # Test the metrics in the evaluation task
    test_func = lambda x: list(map(str, x._eval_tasks[0]._metric_names))

    try:
        assert test_func(output_loop) == ["CrossEntropyLoss", "Accuracy"]
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "lr_schedule_check",
                "expected": ["CrossEntropyLoss", "Accuracy"],
                "got": test_func(output_loop),
            }
        )
        print(
            f"Wrong metrics in evaluations task. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_next_symbol(target, TransformerLM):
    # Generating input for signature initialization

    cur_output_tokens = [13, 483, 320, 4219, 132, 213, 2775, 10, 1, 0]
    token_length = len(cur_output_tokens)
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (
        padded_length - token_length
    )  # @REPLACE padded = cur_output_tokens + None
    padded_with_batch = np.array(padded)[None, :]

    # Initializing the model with the provided signatures
    seed = 1234
    the_model = TransformerLM(mode="eval")
    the_model.init(
        (shapes.signature(padded_with_batch), shapes.signature(padded_with_batch)),
        rng=PRNGKey(seed),
        use_cache=False,
    )

    successful_cases = 0
    failed_cases = []

    next_de_tokens = target(cur_output_tokens, the_model)

    try:
        assert isinstance(next_de_tokens, int)
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "lr_schedule_check", "expected": int, "got": type(next_de_tokens),}
        )
        print(f"Output must be an integer but got {type(next_de_tokens)}")

    try:
        assert next_de_tokens == 23317
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "lr_schedule_check", "expected": 23317, "got": next_de_tokens,}
        )
        print(f"Expected output is 23317. Got {next_de_tokens}.")

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_greedy_decode(target):

    the_model = None
    successful_cases = 0
    failed_cases = []

    test_next_symbol = next_symbol_mocker(
        path_test_files="./test_support_files/i_just_found_roses_not_tulips_test/output.pkl"
    )
    test_sentence = "It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips."

    output = target(test_sentence, the_model, next_symbol=test_next_symbol.mocked_fn)
    expected = ": I just found roses, not tulips.<EOS>"

    try:

        assert output == expected
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "next_symbol_check1", "expected": expected, "got": output,}
        )
        print(
            f"Wrong output. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}"
        )

    test_next_symbol = next_symbol_mocker(
        path_test_files="./test_support_files/four_students_test/output.pkl"
    )
    test_sentence = "It’s the posing craze sweeping the U.S. after being brought to fame by skier Lindsey Vonn, soccer star Omar Cummings, baseball player Albert Pujols - and even Republican politician Rick Perry. But now four students at Riverhead High School on Long Island, New York, have been suspended for dropping to a knee and taking up a prayer pose to mimic Denver Broncos quarterback Tim Tebow. Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were all suspended for one day because the ‘Tebowing’ craze was blocking the hallway and presenting a safety hazard to students. Scroll down for video. Banned: Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll (all pictured left) were all suspended for one day by Riverhead High School on Long Island, New York, for their tribute to Broncos quarterback Tim Tebow. Issue: Four of the pupils were suspended for one day because they allegedly did not heed to warnings that the 'Tebowing' craze at the school was blocking the hallway and presenting a safety hazard to students."
    output = target(test_sentence, the_model, next_symbol=test_next_symbol.mocked_fn)
    expected = "Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were\nsuspended for one day. Four students were suspended for one day\nbecause they allegedly did not heed to warnings that the 'Tebowing'\ncraze was blocking the hallway and presenting a safety hazard to\nstudents.<EOS>"

    try:
        assert output == expected
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "next_symbol_check1", "expected": expected, "got": output,}
        )
        print(
            f"Wrong output. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}"
        )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
