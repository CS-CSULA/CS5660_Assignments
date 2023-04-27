# -*- coding: utf-8 -*-
import numpy as np
import random
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training
from trax import shapes
from jax.random import PRNGKey

import glob
import os
import re
import pickle


VOCAB_FILE = "ende_32k.subword"
VOCAB_DIR = "data/"


def test_input_encoder_fn(target):
    success = 0
    fails = 0

    input_vocab_size = 10
    d_model = 2
    n_encoder_layers = 6

    encoder = target(input_vocab_size, d_model, n_encoder_layers)
    lstms = "\n".join([f"  LSTM_{d_model}"] * n_encoder_layers)

    expected = f"Serial[\n  Embedding_{input_vocab_size}_{d_model}\n{lstms}\n]"

    proposed = str(encoder)

    # Test all layers are in the expected sequence
    try:
        assert proposed.replace(" ", "") == expected.replace(" ", "")
        success += 1
    except:
        fails += 1
        print("Wrong model. \nProposed:\n%s" % proposed, "\nExpected:\n%s" % expected)

    # Test the output type
    try:
        assert isinstance(encoder, trax.layers.combinators.Serial)
        success += 1
        # Test the number of layers
        try:
            # Test
            assert len(encoder.sublayers) == (n_encoder_layers + 1)
            success += 1
        except:
            fails += 1
            print(
                "The number of sublayers does not match %s <>" % len(encoder.sublayers),
                " %s" % (n_encoder_layers + 1),
            )
    except:
        fails += 1
        print("The enconder is not an object of ", trax.layers.combinators.Serial)

    #----------------------------    
    input_vocab_size = 15
    d_model = 5
    n_encoder_layers = 4

    encoder = target(input_vocab_size, d_model, n_encoder_layers)
    lstms = "\n".join([f"  LSTM_{d_model}"] * n_encoder_layers)

    expected = f"Serial[\n  Embedding_{input_vocab_size}_{d_model}\n{lstms}\n]"

    proposed = str(encoder)

    # Test all layers are in the expected sequence
    try:
        assert proposed.replace(" ", "") == expected.replace(" ", "")
        success += 1
    except:
        fails += 1
        print("Wrong model. \nProposed:\n%s" % proposed, "\nExpected:\n%s" % expected)

    # Test the output type
    try:
        assert isinstance(encoder, trax.layers.combinators.Serial)
        success += 1
        # Test the number of layers
        try:
            # Test
            assert len(encoder.sublayers) == (n_encoder_layers + 1)
            success += 1
        except:
            fails += 1
            print(
                "The number of sublayers does not match %s <>" % len(encoder.sublayers),
                " %s" % (n_encoder_layers + 1),
            )
    except:
        fails += 1
        print("The enconder is not an object of ", trax.layers.combinators.Serial)

        
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


def test_pre_attention_decoder_fn(target):
    success = 0
    fails = 0

    mode = "train"
    target_vocab_size = 10
    d_model = 2

    decoder = target(mode, target_vocab_size, d_model)

    expected = f"Serial[\n  Serial[\n    ShiftRight(1)\n  ]\n  Embedding_{target_vocab_size}_{d_model}\n  LSTM_{d_model}\n]"

    proposed = str(decoder)

    # Test all layers are in the expected sequence
    try:
        assert proposed.replace(" ", "") == expected.replace(" ", "")
        success += 1
    except:
        fails += 1
        print("Wrong model. \nProposed:\n%s" % proposed, "\nExpected:\n%s" % expected)

    # Test the output type
    try:
        assert isinstance(decoder, trax.layers.combinators.Serial)
        success += 1
        # Test the number of layers
        try:
            # Test
            assert len(decoder.sublayers) == 3
            success += 1
        except:
            fails += 1
            print(
                "The number of sublayers does not match %s <>" % len(decoder.sublayers),
                " %s" % 3,
            )
    except:
        fails += 1
        print("The enconder is not an object of ", trax.layers.combinators.Serial)

    #----------------------------
    mode = "train"
    target_vocab_size = 20
    d_model = 5

    decoder = target(mode, target_vocab_size, d_model)

    expected = f"Serial[\n  Serial[\n    ShiftRight(1)\n  ]\n  Embedding_{target_vocab_size}_{d_model}\n  LSTM_{d_model}\n]"

    proposed = str(decoder)

    # Test all layers are in the expected sequence
    try:
        assert proposed.replace(" ", "") == expected.replace(" ", "")
        success += 1
    except:
        fails += 1
        print("Wrong model. \nProposed:\n%s" % proposed, "\nExpected:\n%s" % expected)

    # Test the output type
    try:
        assert isinstance(decoder, trax.layers.combinators.Serial)
        success += 1
        # Test the number of layers
        try:
            # Test
            assert len(decoder.sublayers) == 3
            success += 1
        except:
            fails += 1
            print(
                "The number of sublayers does not match %s <>" % len(decoder.sublayers),
                " %s" % 3,
            )
    except:
        fails += 1
        print("The enconder is not an object of ", trax.layers.combinators.Serial)

    
    
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


def test_prepare_attention_input(target):
    success = 0
    fails = 0

    test_cases = [{'name': 'test_1'
                   ,'input': {'encoder_activations': fastnp.array([
                                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]],
                                        ]
                                    ),
                             'decoder_activations': fastnp.array(
                                            [
                                                [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]],
                                                [[2, 0, 2, 0], [0, 2, 0, 2], [0, 0, 0, 0]],
                                            ]
                                        ),
                             'inputs': fastnp.array([[1, 2, 3], [1, 4, 0]])
                            }
                    ,'expected': {'enc_act': fastnp.array([
                                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]],
                                        ]
                                    ),
                             'dec_act': fastnp.array(
                                            [
                                                [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]],
                                                [[2, 0, 2, 0], [0, 2, 0, 2], [0, 0, 0, 0]],
                                            ]
                                        ),
                             'exp_mask': fastnp.array(
                                                    [
                                                        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                                                        [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
                                                    ]
                                                )
                                }
                  }, 
                  {'name': 'test_2'
                   ,'input': {'encoder_activations': fastnp.array(
                                                        [
                                                            [[1, 0, 0,], [0, 1, 0,], [0, 0, 1,]],
                                                            [[1, 0, 1,], [0, 1, 0,], [0, 0, 0,]],
                                                        ]
                                                    )
                             , 'decoder_activations': fastnp.array(
                                                        [
                                                            [[2, 0, 0,], [0, 2, 0,], [0, 0, 2,]],
                                                            [[2, 0, 2,], [0, 2, 0,], [0, 0, 0,]],
                                                        ]
                                                    )
                             , 'inputs': fastnp.array([[1, 2, 3], [1, 4, 0]])
                            }
                    ,'expected': {'enc_act': fastnp.array(
                                                    [
                                                        [[1, 0, 0,], [0, 1, 0,], [0, 0, 1,]],
                                                        [[1, 0, 1,], [0, 1, 0,], [0, 0, 0,]],
                                                    ]
                                                )
                                  , 'dec_act': fastnp.array(
                                                    [
                                                        [[2, 0, 0,], [0, 2, 0,], [0, 0, 2,]],
                                                        [[2, 0, 2,], [0, 2, 0,], [0, 0, 0,]],
                                                    ]
                                                )
                                  , 'exp_mask': fastnp.array(
                                                        [
                                                            [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                                                            [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
                                                        ]
                                                    )
                                }
                  },
                  {'name': 'test_3'
                   ,'input': {'encoder_activations': fastnp.array(
                                                        [
                                                            [[1, 0,0], [0, 1, 0],],
                                                            [[1, 0,1], [0, 1, 0],],
                                                        ]
                                                    )
                             , 'decoder_activations': fastnp.array(
                                                            [
                                                                [[2, 0,0], [0, 2,0],],
                                                                [[2, 0,2], [0, 2,0],],
                                                            ]
                                                        )
                             , 'inputs': fastnp.array([[1, 2, 3], [1, 4, 0]])
                             }
                    ,'expected': {'enc_act': fastnp.array(
                                                    [
                                                        [[1, 0,0], [0, 1, 0],],
                                                        [[1, 0,1], [0, 1, 0],],
                                                    ]
                                                )
                                  , 'dec_act': fastnp.array(
                                                    [
                                                        [[2, 0,0], [0, 2,0],],
                                                        [[2, 0,2], [0, 2,0],],
                                                    ]
                                                )
                                  , 'exp_mask': fastnp.array(
                                                        [
                                                            [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                                                            [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
                                                        ]
                                                    )
                                 }
                  },
                ]    
                
    for test_case in test_cases:
        exp_type = type(test_case['expected']['enc_act'])

        queries, keys, values, mask = target(**test_case['input'])
        
    try:
        assert fastnp.allclose(queries, test_case['expected']['dec_act'])
        success += 1
    except:
        fails += 1
        print("Queries does not match the decoder activations")
    try:
        assert fastnp.allclose(keys, test_case['expected']['enc_act'])
        success += 1
    except:
        fails += 1
        print("Keys does not match the encoder activations")
    try:
        assert fastnp.allclose(values, test_case['expected']['enc_act'])
        success += 1
    except:
        fails += 1
        print("Values does not match the encoder activations")
    try:
        assert fastnp.allclose(mask, test_case['expected']['exp_mask'])
        success += 1
    except:
        fails += 1
        print(
            "Mask does not match expected tensor. \nExpected:\n%s" % test_case['expected']['exp_mask'],
            "\nOutput:\n%s" % mask,
        )

    # Test the output type
    try:
        assert isinstance(queries, exp_type)
        assert isinstance(keys, exp_type)
        assert isinstance(values, exp_type)
        assert isinstance(mask, exp_type)
        success += 1
    except:
        fails += 1
        print(
            "One of the output object are not of type ",
            jax.interpreters.xla.DeviceArray,
        )
        
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")



def test_NMTAttn(target):
    test_cases = [{'name': 'default_test_check'
                   , 'input': {}
                   , 'expected': {'str_rep': "Serial_in2_out2[\n  Select[0,1,0,1]_in2_out4\n  Parallel_in2_out2[\n    Serial[\n      Embedding_33300_1024\n      LSTM_1024\n      LSTM_1024\n    ]\n    Serial[\n      Serial[\n        ShiftRight(1)\n      ]\n      Embedding_33300_1024\n      LSTM_1024\n    ]\n  ]\n  PrepareAttentionInput_in3_out4\n  Serial_in4_out2[\n    Branch_in4_out3[\n      None\n      Serial_in4_out2[\n        _in4_out4\n        Serial_in4_out2[\n          Parallel_in3_out3[\n            Dense_1024\n            Dense_1024\n            Dense_1024\n          ]\n          PureAttention_in4_out2\n          Dense_1024\n        ]\n        _in2_out2\n      ]\n    ]\n    Add_in2\n  ]\n  Select[0,2]_in3_out2\n  LSTM_1024\n  LSTM_1024\n  Dense_33300\n  LogSoftmax\n]"
                                 , 'n_sublayers': 9
                                 , 'selection_layer':  ["Select[0,1,0,1]_in2_out4", "Select[0,2]_in3_out2"]
                                 }
                  },
                  {'name': 'default_test_check'
                   , 'input': {'input_vocab_size':100,
                                'target_vocab_size':100,
                                'd_model':16,
                                'n_encoder_layers':3,
                                'n_decoder_layers':3,
                                'n_attention_heads':2,
                                'attention_dropout':0.01,
                                'mode':'train'}
                   , 'expected': {'str_rep': "Serial_in2_out2[\n  Select[0,1,0,1]_in2_out4\n  Parallel_in2_out2[\n    Serial[\n      Embedding_100_16\n      LSTM_16\n      LSTM_16\n      LSTM_16\n    ]\n    Serial[\n      Serial[\n        ShiftRight(1)\n      ]\n      Embedding_100_16\n      LSTM_16\n    ]\n  ]\n  PrepareAttentionInput_in3_out4\n  Serial_in4_out2[\n    Branch_in4_out3[\n      None\n      Serial_in4_out2[\n        _in4_out4\n        Serial_in4_out2[\n          Parallel_in3_out3[\n            Dense_16\n            Dense_16\n            Dense_16\n          ]\n          PureAttention_in4_out2\n          Dense_16\n        ]\n        _in2_out2\n      ]\n    ]\n    Add_in2\n  ]\n  Select[0,2]_in3_out2\n  LSTM_16\n  LSTM_16\n  LSTM_16\n  Dense_100\n  LogSoftmax\n]"
                                 , 'n_sublayers': 10
                                 , 'selection_layer':  ["Select[0,1,0,1]_in2_out4", "Select[0,2]_in3_out2"]
                                 }
                  },
                 ]
                 
    # f"Serial_in2_out2[\n  Select[0,1,0,1]_in2_out4\n  Parallel_in2_out2[\n    Serial[\n      Embedding_100_16\n      LSTM_16\n      LSTM_16\n      LSTM_16\n    ]\n    Serial[\n      Serial[\n        ShiftRight(1)\n      ]\n      Embedding_100_16\n      LSTM_16\n    ]\n  ]\n  PrepareAttentionInput_in3_out4\n  Serial_in4_out2[\n    Branch_in4_out3[\n      None\n      Serial_in4_out2[\n        _in4_out4\n        Serial_in4_out2[\n          Parallel_in3_out3[\n            Dense_16\n            Dense_16\n            Dense_16\n          ]\n          PureAttention_in4_out2\n          Dense_16\n        ]\n        _in2_out2\n      ]\n    ]\n    Add_in2\n  ]\n  Select[0,2]_in3_out2\n  LSTM_16\n  LSTM_16\n  LSTM_16\n  Dense_100\n  LogSoftmax\n]"
    
    success = 0
    fails = 0

    for test_case in test_cases:
        result = target(**test_case['input'])
        try:            
            assert test_case["expected"]["str_rep"] == str(result)
            success += 1
        except:
            print(f"The NMTAttn model is not defined properly.\n\tExpected {test_case['expected']['str_rep']}.\n\tGot {str(result)}.")            
            fails += 1
            
        try:
            assert test_case["expected"]["n_sublayers"] == len(result.sublayers)
            success += 1
        except:
            print(f"There are {len(result.sublayers)} layers in your model. There should be {test_case['expected']['n_sublayers']}.\nCheck the LSTM stack before the dense layer")
            fails += 1
        
        
        output = [str(result.sublayers[0]), str(result.sublayers[4])]
        check_count = 0

        for i in range(len(test_case['expected']['selection_layer'])):
            if test_case['expected']['selection_layer'][i] != output[i]:
                print(f"There is a problem with your selection layers.\n\t Expected {test_case['expected']['selection_layer'][i]}.\n\tGot {output[i]}.")
                fails += 1
                break
            else:
                check_count += 1

        if check_count == len(test_case['expected']['selection_layer']):
            success += len(test_case['expected']['selection_layer'])
        

    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


def my_gen():
    while True:
        lng1 = random.randrange(16, 64)
        sample1 = np.random.randint(2, 2000, lng1)
        sample1[lng1 - 1] = 1
        lng2 = random.randrange(16, 64)
        sample2 = np.random.randint(2, 2000, lng2)
        sample2[lng2 - 1] = 1
        yield sample1, sample2


# test train_task
def test_train_task(target):

    train_batch_stream_test = trax.data.BucketByLength(
        [64],
        [32],  # boundaries, batch_sizes,
        length_keys=[0, 1],  # As before: count inputs and targets to length.
    )(my_gen())

    train_batch_stream_test = trax.data.AddLossWeights(id_to_mask=0)(
        train_batch_stream_test
    )

    train_task = target(train_batch_stream_test)

    success = 0
    fails = 0

    # Test the labeled data parameter
    try:
        strlabel = str(train_task._labeled_data)
        assert strlabel.find("generator") and strlabel.find("add_loss_weights")
        success += 1
    except:
        fails += 1
        print("Wrong labeled data parameter")

    # Test the cross entropy loss data parameter
    try:
        strlabel = str(train_task._loss_layer)
        assert strlabel == "CrossEntropyLoss_in3"
        success += 1
    except:
        fails += 1
        print("Wrong loss functions. CrossEntropyLoss_in3 was expected")

    # Test the optimizer parameter
    try:
        assert isinstance(train_task.optimizer, trax.optimizers.adam.Adam)
        success += 1
    except:
        fails += 1        
        print("Wrong optimizer")
        
    opt_params_dict = {'weight_decay_rate': fastnp.array(1.e-5),
                         'b1': fastnp.array(0.9),
                         'b2': fastnp.array(0.999),
                         'eps': fastnp.array(1.e-5),
                         'learning_rate': fastnp.array(0.01)}
    
    try: 
        assert train_task._optimizer.opt_params == opt_params_dict
        success += 1
    except:
        fails += 1
        print(f"Optimizer has the wrong parameters.\n\tExpected {opt_params_dict}.\n\tGot {train_task._optimizer.opt_params}.")

    # Test the schedule parameter
    try:
        assert isinstance(
            train_task._lr_schedule, trax.supervised.lr_schedules._BodyAndTail
        )
        success += 1
    except:
        fails += 1
        print("Wrong learning rate schedule type")
        
    try: 
        assert train_task._lr_schedule._body_value == 0.01
        success += 1
    except:
        fails += 1
        print(f'Wrong learning rate value.\n\tExpected 0.01.\n\tGot {train_task._lr_schedule._body_value}.')

    # Test the _n_steps_per_checkpoint parameter
    try:
        assert train_task._n_steps_per_checkpoint == 10
        success += 1
    except:
        fails += 1
        print("Wrong checkpoint step frequency")

    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


def test_next_symbol(target, model):
    # Generating input for signature initialization

    tokens_en = np.array([[17332, 140, 172, 207, 1]])
    cur_output_tokens = []
    token_length = len(cur_output_tokens)
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (
        padded_length - token_length
    )  # @REPLACE padded = cur_output_tokens + None
    padded_with_batch = np.array(padded)[None, :]

    # Initializing the model with the provided signatures
    seed = 12345
    the_model = model(mode="eval")
    the_model.init(
        (shapes.signature(tokens_en), shapes.signature(padded_with_batch)),
        rng=PRNGKey(seed),
        use_cache=False,
    )

    next_de_tokens = target(the_model, tokens_en, [], 0.0)
    assert isinstance(next_de_tokens, tuple), "Output must be a tuple"
    assert len(next_de_tokens) == 2, "Size of tuple must be 2"
    assert type(next_de_tokens[0]) == int and type(next_de_tokens[1]) == float, "Tuple must contain an integer and a float number"
    
    # Test an output
    next_de_tokens = target(the_model, tokens_en, [18477], 0.0)
    # print('next_de_tokens', next_de_tokens)
    assert np.allclose([next_de_tokens[0], next_de_tokens[1]], [7283, -9.929085731506348]), f"Expected output: [{7283}, {-9.929085731506348}], your output: [{next_de_tokens[0]}, {next_de_tokens[1]}]"
    
    print("\033[92m All tests passed")


class next_symbol_mock:
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
        self.dict_in_out_map = self.read_path_test_files()

    def read_path_test_files(self):
        """Reads files in .pkl format.
                
        Returns:
            dict: Dictionary that maps the input and output directly.
        """

        dict_raw_output = {}
        # print(os.getcwd())
        # print(self.path_test_files)
        for filename in glob.glob(os.path.join(self.path_test_files, "*.pkl")):
            filename_number = int(re.findall("[0-9]+", filename)[0])
            # print('filename', filename)
            # print('filename_number', filename_number)
            with open(filename, "rb") as f:
                dict_raw_output.update(pickle.load(f))

        return dict_raw_output

    def mocked_fn(
        self, NMTAttn=None, input_tokens=None, cur_output_tokens=None, temperature=0.0
    ):
        """Returns the input/output mapping using the dictionary that 
        was read in read_path_test_files().

        Args:
            NMTAttn (tl.Serial): Instantiated model. This parameter is not actually used but 
                is left as the learner implementation requires it.
            input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
            cur_output_tokens (list): tokenized representation of previously translated words
            temperature (float): parameter for sampling ranging from 0.0 to 1.0. This parameter 
                is not actually used but is left as the learner implementation requires it.
            
            vocab_file (str): filename of the vocabulary
            vocab_dir (str): path to the vocabulary file

        Returns:
            tuple: (int, float)
                int: index of the next token in the translated sentence
                float: log probability of the next symbol
        """
        return self.dict_in_out_map.get(
            (tuple(map(tuple, input_tokens)), tuple(cur_output_tokens))
        )


# UNIT TEST
# test sampling_decode
def test_sampling_decode(target):

    the_model = None
    success = 0
    fails = 0

    try:
        test_next_symbol = next_symbol_mock(
            path_test_files="./test_support_files/i_eat_soup_test"
        )
        output = target(
            "I eat soup.",
            NMTAttn=the_model,
            temperature=0,
            vocab_file="ende_32k.subword",
            vocab_dir="data/",
            next_symbol=test_next_symbol.mocked_fn,
        )
        # print('output in test 1', output)
        expected = (
            [161, 15103, 5, 25132, 35, 3, 1],
            -0.0003108978271484375,
            "Ich iss Suppe.",
        )
        assert (np.array(output[0]) == np.array(expected[0])).all()
        assert np.isclose(output[1], expected[1]).all()
        assert output[2] == expected[2]
        success += 1
    except:
        fails += 1
        print("Test 1 failed")

    try:
        test_next_symbol = next_symbol_mock(
            path_test_files="./test_support_files/i_like_your_shoes_test"
        )
        output = target(
            "I like your shoes.",
            NMTAttn=the_model,
            temperature=0,
            vocab_file="ende_32k.subword",
            vocab_dir="data/",
            next_symbol=test_next_symbol.mocked_fn,
        )
        # print('output in test 2', output)
        expected = (
            [161, 3383, 607, 18144, 35, 3, 1],
            -0.0006542205810546875,
            "Ich mag Ihre Schuhe.",
        )

        assert (np.array(output[0]) == np.array(expected[0])).all()
        assert np.isclose(output[1], expected[1]).all()
        assert output[2] == expected[2]
        assert output[2] == expected[2]
        success += 1
    except:
        fails += 1
        print("Test 2 failed")

    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


def test_rouge1_similarity(target):

    success = 0
    fails = 0
    n_samples = 10

    test_cases = [
        {
            "name": "simple_test_check",
            "input": [[1, 2, 3], [1, 2, 3, 4]],
            "expected": 0.8571428571428571,
            "error": "Expected similarity: 0.8571428571428571",
        },
        {
            "name": "simple_test_check",
            "input": [[2, 1], [3, 1]],
            "expected": 0.5,
            "error": "Expected similarity: 0.5",
        },
        {
            "name": "simple_test_check",
            "input": [[2], [3]],
            "expected": 0,
            "error": "Expected similarity: 0",
        },
        {
            "name": "simple_test_check",
            "input": [[0] * 100 + [2] * 100, [0] * 100 + [1] * 100],
            "expected": 0.5,
            "error": "Expected similarity: 0.5",
        },
    ]

    private_test_cases = [
        {
            "name": "simple_test_check",
            "input": [[[9, 8, 7, 6, 5], [9, 5, 4]]],
            "expected": 0.5,
            "error": "The output from rouge1_similarity does not match.",
        },
        {
            "name": "simple_test_check",
            "input": [
                [0] * 10 + [2] * 10 + [4] * 10,
                [0] * 11 + [2] * 9 + [4] * 5 + [8] * 6,
            ],
            "expected": 0.7868852459016393,
            "error": "The output from rouge1_similarity does not match.",
        },
        {
            "name": "simple_test_check",
            "input": [[0] * 5 + [1] * 5, [2] * 10],
            "expected": 0,
            "error": "The output from rouge1_similarity does not match.",
        },
    ]

    for test_case in test_cases:

        try:
            if test_case["name"] == "simple_test_check":
                assert abs(test_case["expected"] - target(*test_case["input"])) < 1e-6
                success += 1
        except:
            print(test_case["error"])
            fails += 1

    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


def test_average_overlap(target, rouge1_similarity):
    success = 0
    fails = 0

    test_cases = [
        {
            "name": "dict_test_check",
            "input": [rouge1_similarity, [[1, 2], [3, 4], [1, 2], [3, 5]]],
            "expected": {
                0: 0.3333333333333333,
                1: 0.16666666666666666,
                2: 0.3333333333333333,
                3: 0.16666666666666666,
            },
            "error": "Expected output does not match",
        },
        {
            "name": "dict_test_check",
            "input": [
                rouge1_similarity,
                [[1, 2], [3, 4], [1, 2, 5], [3, 5], [3, 4, 1]],
            ],
            "expected": {
                0: 0.30000000000000004,
                1: 0.325,
                2: 0.38333333333333336,
                3: 0.325,
                4: 0.4833333333333334,
            },
            "error": "Expected output does not match",
        },
    ]

    for test_case in test_cases:
        try:
            if test_case["name"] == "dict_test_check":
                output = target(*test_case["input"])
                for x in output:
                    assert abs(output[x] - test_case["expected"][x]) < 1e-5
                    success += 1
        except:
            print(test_case["error"])
            fails += 1

    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")


class generate_samples_mock:
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
        self.tuple_output = self.read_path_test_files()

    def read_path_test_files(self):
        """Reads files in .pkl format.
                
        Returns:
            dict: Dictionary that maps the input and output directly.
        """

        with open(self.path_test_files, "rb") as f:
            tuple_raw = pickle.load(f)

        return tuple_raw

    def mocked_fn(
        self,
        sentence="",
        n_samples=4,
        NMTAttn=None,
        temperature=0.6,
        vocab_file=None,
        vocab_dir=None,
        sampling_decode=None,
        next_symbol=None,
        tokenize=None,
        detokenize=None,
    ):

        return (self.tuple_output[0], self.tuple_output[1])


# test mbr_decode
def test_mbr_decode(target, score_fn, similarity_fn):
    success = 0
    fails = 0

    TEMPERATURE = 0.6
    VOCAB_FILE = "ende_32k.subword"
    VOCAB_DIR = "data/"

    test_cases = [
        {
            "name": "simple_test_check",
            "path_file": "./test_support_files/generate_sample_mock/generate_samples_I_eat_soup_sample4_temp_dot6.pkl",
            "input": "I eat soup.",
            "expected": "Ich iss Suppe.",
            "error": "Expected output does not match",
        },
        {
            "name": "simple_test_check",
            "path_file": "./test_support_files/generate_sample_mock/generate_samples_I_am_hungry.pkl",
            "input": "I am hungry",
            "expected": "Ich bin hungrig.",
            "error": "Expected output does not match",
        },
        {
            "name": "simple_test_check",
            "path_file": "./test_support_files/generate_sample_mock/generate_samples_Congratulations.pkl",
            "input": "Congratulations!",
            "expected": "Herzlichen Glückwunsch!",
            "error": "Expected output does not match",
        },
        {
            "name": "simple_test_check",
            "path_file": "./test_support_files/generate_sample_mock/generate_samples_You_have_completed_the_assignment.pkl",
            "input": "You have completed the assignment!",
            "expected": "Sie haben die Abtretung abgeschlossen!",
            "error": "Expected output does not match",
        },
    ]

    for test_case in test_cases:
        try:
            test_generate_samples = generate_samples_mock(test_case["path_file"])
            output = target(
                test_case["input"],
                4,
                score_fn=score_fn,
                similarity_fn=similarity_fn,
                NMTAttn=None,
                temperature=TEMPERATURE,
                vocab_file=VOCAB_FILE,
                vocab_dir=VOCAB_DIR,
                generate_samples=test_generate_samples.mocked_fn,
            )
            assert len(output) == 3
            assert output[0] == test_case["expected"]
            success += 1
        except:
            print(test_case["error"])
            fails += 1

    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", success, " Tests passed")
        print("\033[91m", fails, " Tests failed")
