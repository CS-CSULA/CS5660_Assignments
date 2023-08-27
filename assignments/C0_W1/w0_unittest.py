import numpy as np


def test_square(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {"name": "default_check", "input": {"x": 2.11}, "expected": 4.4521},
        {
            "name": "negative_check",
            "input": {"x": -4.92},
            "expected": 24.2064,
        },
        {"name": "larger_pos_check", "input": {"x": 1024}, "expected": 1048576},
        {
            "name": "larger_neg_check",
            "input": {"x": -512},
            "expected": 262144,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output from sigmoid function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_plus_five(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {"name": "default_check", "input": {"x": 2.11}, "expected": 7.11},
        {
            "name": "negative_check",
            "input": {"x": -4.92},
            "expected": 0.08,
        },
        {"name": "larger_pos_check", "input": {"x": 1024}, "expected": 1029},
        {
            "name": "larger_neg_check",
            "input": {"x": -512},
            "expected": -507,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output from sigmoid function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
