#!/usr/bin/env python3
import sys
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from itertools import product
from pathlib import Path
from pickle import load
from pprint import pprint
from unittest import TestCase

import assertions


_TEST_PICKLE_FILE = 'test_assertions.pickle'

# ---- Reference functions, i.e. simulate teacher and student solutions ----


def _reference_printed(x, y=7):
    print(f"{x} + {y} = {x + y}")


def _reference_returned(x, y=7):
    return x * y


def _reference_raised(x, y=0):
    return x // y


def _reference_input(x):
    return x + int(input())


def _reference_debug(x, y=7):
    print('Spam')
    if y is None:
        print('Eggs')
        y = int(input())
    return x + y


def _reference_call_by_reference(a_dict):
    for k1, v in a_dict.items():
        assert isinstance(v, dict)
        for k2, w in v.items():
            assert isinstance(w, int)
            if w < 0:
                v[k2] = 0
            elif w > 100:
                v[k2] = 100
    return a_dict


_test_create_assertion_call_by_reference_args = ({
    'a': {
        'x': -1,
        'y': 50,
        'z': 101,
    },
    'b': {
        'x': -1000,
        'y': 1000,
    }
},)
_test_create_assertion_call_by_reference_return = {
    'a': {
        'x': 0,
        'y': 50,
        'z': 100,
    },
    'b': {
        'x': 0,
        'y': 100,
    }
}


class TestAssertions(TestCase):

    # -------- Test helper functions --------


    @classmethod
    def _delete_pickle(cls):
        pickle_file = Path(_TEST_PICKLE_FILE)
        if pickle_file.exists():
            pickle_file.unlink()

    @classmethod
    def _load_pickle(cls):
        pickle_file = Path(_TEST_PICKLE_FILE)
        if not pickle_file.exists():
            return None
        with pickle_file.open('rb') as file:
            return load(file)

    # ---- Test _call_function() ----

    def test_call_function(self):
        called = False

        def f():
            nonlocal called
            called = True
            return None

        printed, returned, raised = assertions._call_function(f)
        self.assertIsNone(printed)
        self.assertIsNone(returned)
        self.assertIsNone(raised)
        self.assertTrue(called)

    def test_call_function_arguments(self):
        arguments = None

        def f(*args):
            nonlocal arguments
            arguments = args
            return None

        printed, returned, raised = assertions._call_function(f, (1, 2, 3))
        self.assertIsNone(printed)
        self.assertIsNone(returned)
        self.assertIsNone(raised)
        self.assertEqual(arguments, (1, 2, 3))

    def test_call_function_inputs(self):
        inputs = None

        def f():
            nonlocal inputs
            lines = list()
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            inputs = tuple(lines)
            return None

        printed, returned, raised = assertions._call_function(f, None, ('a', 'b', 'c'))
        self.assertIsNone(printed)
        self.assertIsNone(returned)
        self.assertIsNone(raised)
        self.assertEqual(inputs, ('a', 'b', 'c'))

    def test_call_function_print(self):
        def f():
            print('spam')
            print('eggs')

        printed, returned, raised = assertions._call_function(f)
        self.assertEqual(printed, ('spam', 'eggs'))
        self.assertIsNone(returned)
        self.assertIsNone(raised)

    def test_call_function_return(self):
        def f():
            return 42

        printed, returned, raised = assertions._call_function(f)
        self.assertIsNone(printed)
        self.assertEqual(returned, 42)
        self.assertIsNone(raised)

    def test_call_function_raise(self):
        def f():
            raise Exception('Expected exception')

        printed, returned, raised = assertions._call_function(f)
        self.assertIsNone(printed)
        self.assertIsNone(returned)
        self.assertIsNotNone(raised)
        self.assertIsInstance(raised, tuple)
        self.assertEqual(len(raised), 2)
        self.assertEqual(raised[0], Exception)
        self.assertEqual(raised[1], 'Expected exception')

    # ---- Test _clean_output() ----

    def test_clean_output_none(self):
        clean = assertions._clean_output(None)
        self.assertIsNone(clean)

    def test_clean_output_str(self):
        clean = assertions._clean_output(' spam \n eggs ')
        self.assertEqual(clean, ('spam', 'eggs'))

    def test_clean_output_iter(self):
        clean = assertions._clean_output([' spam ', ' eggs '])
        self.assertEqual(clean, ('spam', 'eggs'))

    def test_clean_output_both(self):
        clean = assertions._clean_output([' spam ', None, ' eggs \n both '])
        self.assertEqual(clean, ('spam', 'eggs', 'both'))

    def test_clean_output_hash_str(self):
        clean = assertions._clean_output(' spam \n # eggs ')
        self.assertEqual(clean, ('spam',))

    def test_clean_output_hash_iter(self):
        clean = assertions._clean_output([' spam ', ' # eggs '])
        self.assertEqual(clean, ('spam',))

    def test_clean_output_hash_both(self):
        clean = assertions._clean_output([' spam ', None, ' # eggs \n # both '])
        self.assertEqual(clean, ('spam',))

    # ---- Test _check_printed ----

    def test_check_printed_none_none(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(None, None, subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertIsNone(errors)
        self.assertEqual(stdout, '')
        self.assertEqual(stderr, '')

    def test_check_printed_none_some(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(None, ('spam', 'eggs'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        expect_errors = 'SUBJECT was unexpected'
        self.assertEqual(errors, expect_errors)
        self.assertEqual(stdout, '')
        self.assertEqual(stderr, expect_errors + '\n')

    def test_check_printed_some_none(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('spam', 'eggs'), None, subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        expect_errors = 'SUBJECT was expected, but got nothing'
        self.assertEqual(errors, expect_errors)
        self.assertEqual(stdout, '')
        self.assertEqual(stderr, expect_errors + '\n')

    def test_check_printed_same_same(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('spam', 'eggs'), ('spam', 'eggs'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertIsNone(errors)
        self.assertEqual(stdout, '')
        self.assertEqual(stderr, '')

    def test_check_printed_trail_left(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('spam', 'eggs', 'extra'), ('spam', 'eggs'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertEqual(errors, 'SUBJECT "extra" was expected')
        self.assertEqual(stderr, 'Printed:\nspam\neggs\n')
        self.assertEqual(stdout, '\nExpected:\nspam\neggs\nextra\n\n')

    def test_check_printed_trail_right(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('spam', 'eggs'), ('spam', 'eggs', 'extra'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertEqual(errors, 'SUBJECT "extra" was unexpected')
        self.assertEqual(stderr, 'Printed:\nspam\neggs\nextra\n')
        self.assertEqual(stdout, '\nExpected:\nspam\neggs\n\n')

    def test_check_printed_both(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('spam', 'eggs'), ('eggs', 'spam'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertEqual(errors, '\n'.join((
            'SUBJECT "eggs" doesn\'t match expected string "spam"',
            'SUBJECT "spam" doesn\'t match expected string "eggs"'
        )))
        self.assertEqual(stderr, 'Printed:\neggs\nspam\n')
        self.assertEqual(stdout, '\nExpected:\nspam\neggs\n\n')

    def test_check_printed_int_right(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('1', '2'), ('1', '2'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertIsNone(errors)
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, '')

    def test_check_printed_int_wrong(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('1', '2'), ('2', '1'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertEqual(errors, '\n'.join((
            'SUBJECT was 2 but we expected 1',
            'SUBJECT was 1 but we expected 2'
        )))
        self.assertEqual(stderr, 'Printed:\n2\n1\n')
        self.assertEqual(stdout, '\nExpected:\n1\n2\n\n')

    def test_check_printed_float_right(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('0.1', '0.2'), ('0.10000001', '0.20000001'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertIsNone(errors)
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, '')

    def test_check_printed_float_wrong(self):
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            errors = assertions._check_printed(('0.1', '0.2'), ('0.20000001', '0.10000001'), subject='SUBJECT')
            stdout = out.getvalue()
            stderr = err.getvalue()
        self.assertEqual(errors, '\n'.join((
            'SUBJECT was 0.20000001 but we expected 0.1',
            'SUBJECT was 0.10000001 but we expected 0.2'
        )))
        self.assertEqual(stderr, 'Printed:\n0.20000001\n0.10000001\n')
        self.assertEqual(stdout, '\nExpected:\n0.1\n0.2\n\n')

    # ---- Test _check_returned() with check_value=True ----

    def test_check_returned_depth(self):
        errors = assertions._check_returned(None, None, subject='SUBJECT', _depth=0)
        self.assertEqual(errors, f'SUBJECT reached maximum depth of {assertions.MAX_DEPTH}')

    def test_check_returned_none_none(self):
        errors = assertions._check_returned(None, None, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_none_some(self):
        errors = assertions._check_returned(None, 'some', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT was str \'some\' but None was expected')

    def test_check_returned_some_none(self):
        errors = assertions._check_returned('some', None, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT was None but str \'some\' was expected')

    def test_check_returned_type_mismatch_int_str(self):
        errors = assertions._check_returned(42, '42', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT was str \'42\' but a value of type int was expected')

    def test_check_returned_type_mismatch_list_tuple(self):
        errors = assertions._check_returned([1, 2], (1, 2), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT was tuple (1, 2) but a value of type list was expected')

    def test_check_returned_type_mismatch_dict_set(self):
        errors = assertions._check_returned({1: 2, 3: 4}, {1, 3}, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT was set {1, 3} but a value of type dict was expected')

    def test_check_returned_bool_right(self):
        errors = assertions._check_returned(True, True, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_bool_wrong(self):
        errors = assertions._check_returned(False, True, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT bool True is unequal to expected bool False')

    def test_check_returned_int_right(self):
        errors = assertions._check_returned(42, 42, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_int_wrong(self):
        errors = assertions._check_returned(42, 13, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT int 13 is unequal to expected int 42')

    def test_check_returned_float_right(self):
        errors = assertions._check_returned(1.0, 1.0000001, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_float_wrong(self):
        errors = assertions._check_returned(1.0, 2.0000001, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT float 2.0000001 is unequal to expected float 1.0')

    def test_check_returned_str_right(self):
        errors = assertions._check_returned('spam', 'spam', subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_str_wrong(self):
        errors = assertions._check_returned('spam', 'eggs', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "eggs" doesn\'t match expected string "spam"')

    def test_check_returned_regex_right(self):
        errors = assertions._check_returned('/spa+m/i', 'spaaam', subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_regex_wrong(self):
        errors = assertions._check_returned('/spa+m/', 'eggs', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "eggs" doesn\'t match the regular expression "/spa+m/"')

    def test_check_returned_dict_right(self):
        errors = assertions._check_returned({1: 2, 3: 4}, {1: 2, 3: 4}, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_dict_right_deep(self):
        errors = assertions._check_returned({1: {2: 3, 4: {5: 6}}}, {1: {2: 3, 4: {5: 6}}}, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_dict_wrong_deep(self):
        errors = assertions._check_returned({1: {2: 3, 4: {5: 6}}}, {1: {2: 3, 4: {5: 7}}}, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT dict expected to contain {1: {2: 3, 4: {5: 6}}}, but got {1: {2: 3, 4: {5: 7}}}')

    def test_check_returned_dict_keys(self):
        errors = assertions._check_returned({1: 2, 3: 4}, {1: 2}, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT dict {1: 2} expected to contain key(s) 1, 3')

    def test_check_returned_tuple_length(self):
        errors = assertions._check_returned((1, 2), (3, ), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT tuple (3,) expected to contain 2 items, but got 1')

    def test_check_returned_tuple_right(self):
        errors = assertions._check_returned((1, 2, 3), (1, 2, 3), subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_tuple_wrong(self):
        errors = assertions._check_returned((1, 2, 3), (1, 4, 3), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT tuple (1, 4, 3) at index 1: int 4 is unequal to expected int 2')

    def test_check_returned_tuple_right_deep(self):
        errors = assertions._check_returned((1, 2, (3, 4)), (1, 2, (3, 4)), subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_tuple_wrong_deep(self):
        errors = assertions._check_returned((1, 2, (3, 4)), (1, 2, (3, 5)), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT tuple (1, 2, (3, 5)) at index 2: tuple (3, 5) at index 1: int 5 is unequal to expected int 4')

    # ---- Test _check_returned() with check_value=False ----

    def test_check_returned_type_only_simple(self):
        errors = assertions._check_returned(42, 13, check_value=False, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_type_only_list(self):
        errors = assertions._check_returned([1, 2], [3, 4], check_value=False, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_type_only_dict_wrong(self):
        errors = assertions._check_returned({1: 2, 3: 4}, {4: 5, 6: 7}, check_value=False, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT dict {4: 5, 6: 7} expected to contain key(s) 1, 3')

    def test_check_returned_type_only_dict_right(self):
        errors = assertions._check_returned({1: 2, 3: 4}, {1: 5, 3: 7}, check_value=False, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_returned_type_only_dict_deep_wrong(self):
        errors = assertions._check_returned({1: 2, 3: {4: 5, 6: {7: 8}}}, {1: 2, 3: {4: 5, 6: {8: 9}}}, check_value=False, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT dict {1: 2, 3: {4: 5, 6: {8: 9}}} at key 3: dict {4: 5, 6: {8: 9}} at key 6: dict {8: 9} expected to contain key(s) 7')

    def test_check_returned_type_only_dict_deep_right(self):
        errors = assertions._check_returned({1: 2, 3: {4: 5, 6: {7: 8}}}, {1: 3, 3: {4: 6, 6: {7: 9}}}, check_value=False, subject='SUBJECT')
        self.assertIsNone(errors)

    # ---- Test _check_raised() ----

    def test_check_raised_none_none(self):
        errors = assertions._check_raised(None, None, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_raised_none_some(self):
        errors = assertions._check_raised(None, Exception('Expected exception'), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT Exception was raised but not expected: Expected exception')

    def test_check_raised_some_none(self):
        errors = assertions._check_raised(Exception('Expected exception'), None, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT Exception was expected but not raised')

    def test_check_raised_wrong_type(self):
        errors = assertions._check_raised(Exception('Expected exception'), TypeError('Unexpected exception'), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT TypeError was raised, but Exception was expected')

    def test_check_raised_wrong_text(self):
        errors = assertions._check_raised(Exception('Expected exception'), Exception('Unexpected exception'), subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "Unexpected exception" doesn\'t match expected string "Expected exception"')

    def test_check_raised_right(self):
        errors = assertions._check_raised(Exception('Expected exception'), Exception('Expected exception'), subject='SUBJECT')
        self.assertIsNone(errors)

    # ---- Test _check_pattern() ----

    def test_check_pattern_none_none(self):
        errors = assertions._check_pattern(None, None, subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_pattern_none_str(self):
        errors = assertions._check_pattern(None, 'plain', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "plain" was unexpected')

    def test_check_pattern_str_none(self):
        errors = assertions._check_pattern('plain', None, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT expected a string matching "plain", but got None')

    def test_check_pattern_str_wrong(self):
        errors = assertions._check_pattern('rain', 'spain', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "spain" doesn\'t match expected string "rain"')

    def test_check_pattern_str_case(self):
        errors = assertions._check_pattern('plain', 'Plain', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "Plain" doesn\'t match expected string "plain"')

    def test_check_pattern_str_right(self):
        errors = assertions._check_pattern('plain', 'plain', subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_pattern_rex_none(self):
        errors = assertions._check_pattern('/plain/', None, subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT expected to match the regular expression "/plain/", but got None')

    def test_check_pattern_rex_wrong(self):
        errors = assertions._check_pattern('/rain/', 'spain', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "spain" doesn\'t match the regular expression "/rain/"')

    def test_check_pattern_rex_wrong_case(self):
        errors = assertions._check_pattern('/plain/', 'Plain', subject='SUBJECT')
        self.assertEqual(errors, 'SUBJECT "Plain" doesn\'t match the regular expression "/plain/"')

    def test_check_pattern_rex_right_case(self):
        errors = assertions._check_pattern('/plain/i', 'Plain', subject='SUBJECT')
        self.assertIsNone(errors)

    def test_check_pattern_rex_right(self):
        errors = assertions._check_pattern('/pl(ai)+n/', 'plaiaiain', subject='SUBJECT')
        self.assertIsNone(errors)

    # ---- Test _check_type() ----

    def test_get_type_out_of_depth_simple(self):
        for args in product((None, 'spam', object()), (-1, 0)):
            with self.assertRaises(RuntimeError) as cm:
                assertions._get_type(*args)
            self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_out_of_depth_dict(self):
        with self.assertRaises(RuntimeError) as cm:
            assertions._get_type({1: 2, 3: 4}, 1)
        self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_out_of_depth_deep_dict(self):
        with self.assertRaises(RuntimeError) as cm:
            assertions._get_type({1: 2, 3: {4: 5, 6: 7}}, 2)
        self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_out_of_depth_deeper_dict(self):
        with self.assertRaises(RuntimeError) as cm:
            assertions._get_type({1: 2, 3: {4: 5, 6: {7: 8}}}, 3)
        self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_out_of_depth_tuple(self):
        with self.assertRaises(RuntimeError) as cm:
            assertions._get_type((1, 2), 1)
        self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_out_of_depth_deep_tuple(self):
        with self.assertRaises(RuntimeError) as cm:
            assertions._get_type((1, 2, (3, 4)), 2)
        self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_out_of_depth_deeper_tuple(self):
        with self.assertRaises(RuntimeError) as cm:
            assertions._get_type((1, 2, (3, 4, (5, 6))), 3)
        self.assertEqual(str(cm.exception), f'_serialize reached maximum depth of {assertions.MAX_DEPTH}')

    def test_get_type_check_type_simple(self):
        for args in product((None, 'spam', object()), (1, 2, 3)):
            returned = assertions._get_type(*args)
            self.assertEqual(returned, type(args[0]))

    def test_get_type_check_type_dict(self):
        returned = assertions._get_type({1: 2, 3: 4}, 2)
        self.assertEqual(returned, {1: int, 3: int})

    def test_get_type_check_type_dict_deep(self):
        returned = assertions._get_type({1: 2, 3: {4: 5, 6: 7}}, 3)
        self.assertEqual(returned, {1: int, 3: {4: int, 6: int}})

    def test_get_type_check_type_dict_deeper(self):
        returned = assertions._get_type({1: 2, 3: {4: 5, 6: {7: 8}}}, 4)
        self.assertEqual(returned, {1: int, 3: {4: int, 6: {7: int}}})

    # ---- Test create_assertion() ----

    def test_create_assertion_reference_type(self):
        for reference in None, 'spam', object():
            with self.assertRaises(AssertionError) as cm:
                assertions.create_assertion(reference, ())
            self.assertEqual(str(cm.exception), f'Teacher bug; reference must be a function, got {type(reference)}')

    def test_create_assertion_expectations_type_simple(self):
        for expectations in None, 'spam', object():
            with self.assertRaises(AssertionError) as cm:
                assertions.create_assertion(_reference_printed, expectations)
            self.assertEqual(str(cm.exception), f'Teacher bug; expectations must be a list or tuple, got {type(expectations)}')

    def test_create_assertion_expectations_type_nested(self):
        for expectations in [1], (4,):
            with self.assertRaises(AssertionError) as cm:
                assertions.create_assertion(_reference_printed, expectations)
            self.assertEqual(str(cm.exception), f'Teacher bug; expectation should be dictionary, not {type(expectations[0])}')

    def test_create_assertion_printed(self):
        self._delete_pickle()

        assertions.create_assertion(_reference_printed, (
            {
                assertions.ARGUMENTS: 6,
                assertions.EXPECT_PRINTED: '6 + 7 = 13'
            },
            {
                assertions.ARGUMENTS: (6, 36),
                assertions.EXPECT_PRINTED: '6 + 36 = 42'
            }
        ), assertion_file=_TEST_PICKLE_FILE, _debug=False)

        pickled = self._load_pickle()
        self.assertIsInstance(pickled, dict)
        self.assertDictEqual(pickled, {
            '_reference_printed': [
                {
                    'arguments': (6,),
                    'expect_printed': ('6 + 7 = 13',)
                 }, {
                    'arguments': (6, 36),
                    'expect_printed': ('6 + 36 = 42',)
                 }
            ]
        })

        self._delete_pickle()


    def test_create_assertion_returned(self):
        self._delete_pickle()

        assertions.create_assertion(_reference_returned, (
            {
                assertions.ARGUMENTS: 6,
                assertions.EXPECT_RETURNED: 42
            },
            {
                assertions.ARGUMENTS: (6, 36),
                assertions.EXPECT_RETURNED: 216
            }
        ), assertion_file=_TEST_PICKLE_FILE, _debug=False)

        pickled = self._load_pickle()
        self.assertIsInstance(pickled, dict)
        self.assertDictEqual(pickled, {
            '_reference_returned': [
                {
                    'arguments': (6,),
                    'expect_returned': 42
                 }, {
                    'arguments': (6, 36),
                    'expect_returned': 216
                 }
            ]
        })

        self._delete_pickle()



    def test_create_assertion_debug(self):
        self._delete_pickle()
        with StringIO() as out, StringIO() as err, \
             redirect_stdout(out), redirect_stderr(err):
            assertions.create_assertion(_reference_debug, (
                {
                    assertions.ARGUMENTS: 6,
                    assertions.EXPECT_PRINTED: 'Spam',
                    assertions.EXPECT_RETURNED: 13
                },
                {
                    assertions.ARGUMENTS: (6, 7),
                    assertions.EXPECT_PRINTED: ('Spam',),
                    assertions.EXPECT_RETURNED: 13
                },
                {
                    assertions.ARGUMENTS: (6, None),
                    assertions.EXPECT_INPUTS: '7',
                    assertions.EXPECT_PRINTED: ('Spam', 'Eggs'),
                    assertions.EXPECT_RETURNED: 13
                },
                {
                    assertions.ARGUMENTS: (6, '7'),
                    assertions.EXPECT_PRINTED: ('Spam',),
                    assertions.EXPECT_RAISED: TypeError("unsupported operand type(s) for +: 'int' and 'str'")
                },
                {
                    assertions.ARGUMENTS: (6, '7'),
                    assertions.EXPECT_PRINTED: ('Spam',),
                    assertions.EXPECT_RAISED: (TypeError, "unsupported operand type(s) for +: 'int' and 'str'")
                },
                {
                    assertions.ARGUMENTS: (6, '7'),
                    assertions.EXPECT_PRINTED: ('Spam',),
                    assertions.EXPECT_RAISED: TypeError
                },
                {
                    assertions.ARGUMENTS: (6, '7'),
                    assertions.EXPECT_PRINTED: ('Spam',),
                    assertions.EXPECT_RAISED: "unsupported operand type(s) for +: 'int' and 'str'"
                }
            ), assertion_file=_TEST_PICKLE_FILE, _debug=True)

            stdout = out.getvalue()
            stderr = err.getvalue()

        self.assertEqual(stdout, """\
[{'arguments': (6,), 'expect_printed': ('Spam',), 'expect_returned': 13},
 {'arguments': (6, 7), 'expect_printed': ('Spam',), 'expect_returned': 13},
 {'arguments': (6, None),
  'expect_inputs': ('7',),
  'expect_printed': ('Spam', 'Eggs'),
  'expect_returned': 13},
 {'arguments': (6, '7'),
  'expect_printed': ('Spam',),
  'expect_raised': (<class 'TypeError'>,
                    "unsupported operand type(s) for +: 'int' and 'str'")},
 {'arguments': (6, '7'),
  'expect_printed': ('Spam',),
  'expect_raised': (<class 'TypeError'>,
                    "unsupported operand type(s) for +: 'int' and 'str'")},
 {'arguments': (6, '7'),
  'expect_printed': ('Spam',),
  'expect_raised': (<class 'TypeError'>, None)},
 {'arguments': (6, '7'),
  'expect_printed': ('Spam',),
  'expect_raised': (None,
                    "unsupported operand type(s) for +: 'int' and 'str'")}]
""")
        self.assertEqual(stderr, '')




    def test_create_assertion_call_by_reference_proper(self):
        self._delete_pickle()
        self.maxDiff = 2048

        assertions.create_assertion(
            _reference_call_by_reference,
            [
                {
                    assertions.ARGUMENTS: _test_create_assertion_call_by_reference_args,
                    assertions.EXPECT_RETURNED: _test_create_assertion_call_by_reference_return
                }
            ],
            assertion_file=_TEST_PICKLE_FILE,
            _debug=False
        )

        pickled = self._load_pickle()
        self.assertIsInstance(pickled, dict)
        self.assertDictEqual(pickled, {
            '_reference_call_by_reference': [
                {
                    'arguments': _test_create_assertion_call_by_reference_args,
                    'expect_returned': _test_create_assertion_call_by_reference_return
                }
            ]
        })

        self._delete_pickle()


    def test_create_assertion_call_by_reference_lazy(self):
        self._delete_pickle()
        self.maxDiff = 2048

        with StringIO() as out, StringIO() as err, \
                redirect_stdout(out), redirect_stderr(err):
            assertions.create_assertion(
                _reference_call_by_reference,
                [
                    {
                        assertions.ARGUMENTS: _test_create_assertion_call_by_reference_args
                    }
                ],
                assertion_file=_TEST_PICKLE_FILE,
                _debug=False,
                _lazy=True
            )
            stdout = out.getvalue()
            stderr = err.getvalue()

        self.assertEqual(stdout, 'Warning, your reference implementation will NOT be tested!\n')
        self.assertEqual(stderr, '')
        pickled = self._load_pickle()
        self.assertIsInstance(pickled, dict)
        self.assertDictEqual(pickled, {
            '_reference_call_by_reference': [
                {
                    'arguments': _test_create_assertion_call_by_reference_args,
                    'expect_returned': _test_create_assertion_call_by_reference_return
                }
            ]
        })

        self._delete_pickle()


if __name__ == '__main__':
    from unittest import main
    main()
