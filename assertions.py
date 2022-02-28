import io
import pickle
import re
import sys
import types

from contextlib import redirect_stdout
from itertools import zip_longest

# Constants
assertion_file = 'assertions.pickle'
student_user = True
MAX_DEPTH = 20
ARGUMENTS = 'arguments'
EXPECT_PRINTED = 'expect_printed'
EXPECT_RETURNED = 'expect_returned'
EXPECT_RAISED = 'expect_raised'
EXPECT_INPUTS = 'expect_inputs'
CHECK_VALUE = 'check_value'


def create_assertion(reference, expectations, label=None):
    """Run teacher's function, check expectations, and store in assertions.pickle"""

    # Validate function arguments
    assert type(reference) is types.FunctionType, \
        f'Teacher bug; reference must be a function, got {type(reference)}'
    
    assert type(expectations) is list, \
        f'Teacher bug; expectations must be a list, got {type(expectations)}'

    label = reference.__name__  # Ignore argument, always use function name

    save_expectations = list()

    for expectation in expectations:

        # Validate whether expectation is dictionary
        assert isinstance(expectation, dict), \
            f'Teacher bug; expectation should be dictionary, not {type(expectation)}'

        # Validate arguments, coerce it to tuple if necessary
        arguments = expectation.get(ARGUMENTS)
        try:
            arguments = tuple(arguments)
        except TypeError:
            # Not an iterable, make it so
            arguments = (arguments,)

        # Validate expected printed lines, cleaned up and stripped
        expect_printed = _clean_output(expectation.get(EXPECT_PRINTED))

        # Validate expected return value
        expect_returned = expectation.get(EXPECT_RETURNED)

        # Validate expected exception type, str, or (type, str)
        expect_raised = expectation.get(EXPECT_RAISED)
        if type(expect_raised) is str:
            expect_raised = None, expect_raised
        elif type(expect_raised) is type:
            expect_raised = expect_raised.__name__, None
        elif type(expect_raised) is tuple \
        and tuple(type(r) for r in expect_raised) == (type, str):
            expect_raised = expect_raised[0].__name__, expect_raised[1]
        else:
            assert expect_raised is None, \
            f'Teacher bug; exception must be type, str or (type, str), got "{expect_raised}"'

        # Validate expected inputs
        expect_inputs = expectation.get(EXPECT_INPUTS)
        if expect_inputs is not None:
            assert type(expect_inputs) in {list, tuple}, \
            f'Teacher bug; inputs must be list or tuple, got "{expect_inputs}"'
            expect_inputs = tuple(str(i) for i in expect_inputs)

        # Validate check_value
        check_value = expectation.get(CHECK_VALUE)
        if check_value is None:
            check_value = True

        # Call the reference function with the current arguments
        printed, returned, raised = _call_function(reference, arguments, expect_inputs)

        # Check printed lines
        error_printed = _check_printed(expect_printed, printed)
        assert error_printed is None, 'Teacher bug; printed output did not match!'

        # Check returned value
        error_returned = _check_returned(expect_returned, returned, check_value=check_value)
        assert error_returned is None, f'Teacher bug; {error_returned}'

        # Check raised exception
        error_raised = _check_raised(expect_raised,  raised)
        assert error_raised is None, f'Teacher bug; {error_raised}'

        # Reference function behaves as expected, add to our saved cases.
        # If both expected return value and expected exception are None, skip them
        expectation = {
            ARGUMENTS: arguments,
            EXPECT_PRINTED: expect_printed
        }
        if expect_returned is not None or expect_raised is not None or expect_inputs is not None:
            expect_returned = _serialize(expect_returned, check_value=check_value)
            expectation.update({
                EXPECT_RETURNED: expect_returned,
                EXPECT_RAISED: expect_raised,
                EXPECT_INPUTS: expect_inputs,
                CHECK_VALUE: check_value
            })
        save_expectations.append(expectation)

        # Teacher sanity, visually check the generated assertions
        print(save_expectations)
        # Load the previously created assertions from file
        try:
            with open(assertion_file, 'rb') as reader:
                assertions = pickle.load(reader)
        except FileNotFoundError:
            assertions = dict()
        except Exception as exc:
            assert False, f'Teacher bug; unhandled {type(exc).__name__}: {exc}'
        # Merge in the new expectations
        assertions[label] = save_expectations
        # Save the merged expectations to file
        try:
            with open(assertion_file, 'wb') as writer:
                pickle.dump(assertions, writer)
        except Exception as exc:
            assert False, f'Teacher bug; unhandled {type(exc).__name__}: {exc}'


def check_assertion(assignment, label=None, fail_fast=True):
    """Run student's function, check expected behaviours, and report the findings"""

    # Validate function arguments
    assert type(assignment) is types.FunctionType, \
    f'Teacher bug; assignment must be a function, got {type(assignment)}'
    
    label = assignment.__name__ # Ignore argument, always use function name

    # Load the assertions file
    try:
        with open(assertion_file, 'rb') as fp:
            assertions = pickle.load(fp)
    except FileNotFoundError:
        assert False, f'Teacher bug; assertion file not found: {assertion_file}'
    except Exception as exc:
        assert False, f'Teacher bug; unhandled {type(exc).__name__}: {exc}'
    
    # Select the arguments and expectations for the given label
    try:
        expectations = assertions[label]
    except KeyError:
        assert False, \
        f'Teacher bug; no expectations for "{label}", did you run create_assertion() for this label?'
    
    # Run through all our test cases, keep track of total errors
    total_errors = 0
    for expectation in expectations:
        
        # Unpack the expectation
        arguments = expectation.get(ARGUMENTS)
        expect_printed = expectation.get(EXPECT_PRINTED)
        expect_returned = expectation.get(EXPECT_RETURNED)
        expect_raised = expectation.get(EXPECT_RAISED)
        expect_inputs = expectation.get(EXPECT_INPUTS)
        check_value = expectation.get(CHECK_VALUE)

        # Count errors in this test case
        errors = 0
        
        # Call the assignment function with the current arguments
        arguments_string = ", ".join(
            f'\'{arg}\'' if isinstance(arg, str) else str(arg)
            for arg in arguments
        )
        _print(f'Call {assignment.__name__}({arguments_string})', end='')
        if expect_inputs is not None:
            _print(f'\n  with inputs {expect_inputs}', end='')
        _print(' ...', end='')
        printed, returned, raised = _call_function(assignment, arguments, expect_inputs)
        returned = _serialize(returned, check_value=check_value)

        # Check raised exception
        error_raised = _check_raised(expect_raised,  raised)
        if error_raised is not None:
            errors += 1
            _print()
            _error(error_raised)
            _print()

        # Check returned value
        error_returned = _check_returned(expect_returned, returned, check_value=check_value)
        if error_returned is not None:
            errors += 1
            _print()
            _error(error_returned)
            _print()

        # Check printed lines
        error_printed = _check_printed(expect_printed, printed)
        if error_printed is not None:
            errors += 1
            _print()

        if errors == 0:
            _print(' OK')
        else:
            total_errors += errors
            if fail_fast:
                break

    # Raise assertion error if things went wrong, needed for nbgrader, or report success
    assert total_errors == 0, 'Unfortunately you seem to have made one or more mistakes!' 
    _print(f'\nWell done! You seem to have solved {label}!')


# ---- Utility functions ----

def _call_function(function, arguments, inputs=None):
    """Call function with given arguments, capture output, return value or exception"""

    returned = None
    raised = None

    orig_input = None
    if inputs is not None:
        input_index = 0

        def _alt_input(_):
            nonlocal input_index
            if input_index < len(inputs):
                input_index += 1
                return inputs[input_index - 1]
            return ''
        orig_input = __builtins__['input']
        __builtins__['input'] = _alt_input

    with io.StringIO() as buffer, redirect_stdout(buffer):
        try:
            returned = function(*arguments)
        except Exception as exc:
            raised = exc
        printed = _clean_output(buffer.getvalue())

    if orig_input is not None:
        __builtins__['input'] = orig_input

    return printed, returned, raised


def _clean_output(printed):
    """Splits the output in lines, trims each of them, and filters out empty ones."""

    if type(printed) is str:
        printed = printed.split('\n')
    else:
        try:
            printed = (str(p) for p in printed)
        except TypeError:
            printed = (str(printed), )
    
    printed = (line.strip() for line in printed)
    printed = (line for line in printed if len(line) > 0 and line[0] != '#')
    return tuple(printed)


def _check_printed(expect, test, *, subject='Printed output'):
    """Check if the printed lines matches the expectation strings and/or regexes"""

    errors = []
    for pattern, line in zip_longest(expect, test):
        if pattern is None:
            if test is None:
                return None
            errors.append(f'{subject} "{test}" was unexpected')
            continue

        expect_type = str
        expect_parsed = None

        # Look at the expected value for indications of an expected type
        for try_type in (int, float):
            try:
                expect_parsed = try_type(pattern)
                expect_type = try_type
                break
            except (ValueError, TypeError):
                pass

        if expect_type is not str:
            # Special case handling for int and float lines, in particular to
            # avoid FPU artefacts such as 0.1 + 0.2 = 0.30000000000000004
            try:
                parsed = expect_type(line)
            except (ValueError, TypeError):
                errors.append(f'{subject} was {line} but we expected {pattern}')
                continue
            if expect_type is int:
                if parsed != expect_parsed:
                    errors.append(f'{subject} was {line} but we expected {pattern}')
                continue
            if expect_type is float:
                if f'{expect_parsed:.6f}' != f'{parsed:.6f}':
                    errors.append(f'{subject} was {line} but we expected {pattern}')
                continue
            assert False, f'Teacher bug; unhandled typed output case {expect_type}'

        error = _check_pattern(pattern, line, subject=subject)
        if error is not None:
            errors.append(error)

    if len(errors) > 0:
        for which, lines, where in (
            ('Printed', test, sys.stderr),
            ('Expected', expect, sys.stdout)
        ):
            print(f'{which}:', file=where)
            
            if len(lines) == 0:
                print('<empty>', file=where)
            for line in lines:
                if type(line) is tuple:
                    line = line[0]
                print(line, file=where)
            where.flush()

            print(file=sys.stdout, flush=True)

        return '\n'.join(errors)


def _check_returned(expect, test, *, subject='Returned value', check_value=True, _depth=MAX_DEPTH):
    """Check that the return value matches the expected value"""

    if _depth <= 0:
        return f'{subject} reached maximum depth of {MAX_DEPTH}'

    expect_type = type(expect)
    if not isinstance(test, expect_type):
        return f'{subject} was {type(test).__name__} {test}' \
               + f' but a value of type {expect_type} was expected' 
    if expect_type is str:
        return _check_pattern(expect, test, subject=subject)
    if expect_type is float:
        if f'{expect:.6f}' != f'{test:.6f}':
            return f'{subject} {type(test).__name__} {test} is unequal' \
                   + f' to expected {expect_type.__name__} {expect}'
        else:
            return None

    if _is_iterable(expect):
        if not _is_iterable(test):
            return f'{subject} {type(test).__name__} {test} expected to be iterable'
        
        if _is_dict_like(expect):
            if not _is_dict_like(test):
                return f'{subject} {type(test).__name__} {test} expected to be dict-like'
            indices = sorted(list(expect.keys()))
            if indices != sorted(list(test.keys())):
                return f'{subject} {type(test).__name__} {test} expected to contain keys ' \
                       + ', '.join(f'{idx!r}' for idx in indices)
            expect_list = list(expect.values())
            test_list = list(test.values())

        else:
            indices = list(range(len(expect)))
            expect_list = list(expect)
            test_list = list(test)
        
        expect_len = len(expect_list)
        test_len = len(test_list)
        if expect_len != test_len:
            return f'{subject} {type(test).__name__} {test} expected to contain' \
                 f' {expect_len} items, but got {test_len}'

        for i, expect_item, test_item in zip(indices, expect_list, test_list):
            msg = _check_returned(expect_item, test_item, subject=subject, check_value=check_value, _depth=_depth - 1)
            if msg:
                return f'{subject} {type(test).__name__} {test} at index {i!r}: {msg}'

    elif check_value and expect != test:
        return f'{subject} {type(test).__name__} {test} is unequal' \
               + f' to expected {expect_type.__name__} {expect}'


def _check_raised(expect, test, *, subject='Exception'):
    """Check that the raised exception, if any, matches the expected exception"""

    if expect is None and test is not None:
        return f'{subject} {type(test).__name__} was raised but not expected: {str(test)}'
    
    elif expect is not None and test is None:
        return f'{subject} {expect[0]} was expected but not raised'
    
    elif test is not None:  # and expect is not None
        expect_type, expect_pattern = expect
        
        error = None
        if expect_type is not None:
            if expect_type != type(test).__name__:
                error = f'{subject} {type(test).__name__} was raised,' \
                        + f' but {expect_type} was expected'
        elif expect_pattern is not None:
            error = _check_pattern(expect_pattern, str(test), subject=subject)

        return error


def _check_pattern(expect, test, subject=''):
    """Check if given value matches the expectation string or regex"""

    if expect is None:
        if test is None:
            return None
        return f'{subject} "{test}" was unexpected'
    else:
        expect = _make_pattern(expect)
        if type(expect) is str:
            if test is None:
                return f'{subject} " was expected to include the string "{expect}"'
            if expect != test:
                return f'{subject} "{test}" doesn\'t match expected string "{expect}"'
        else:
            expect, regex = expect
            if test is None:
                return f'{subject} " was expected to include something to match "{expect}"'
            if regex.match(test) is None:
                return f'{subject} "{test}" doesn\'t match expected regex "{expect}"'


# Regular expression to match regular expressions
_meta_rex = re.compile(r'^/(.*)(?<!\\)/([ais]*)$')


def _make_pattern(pattern):
    """If the argument matches the meta regex, compile it"""

    match = _meta_rex.match(pattern)
    if match is None:
        # Not a regex
        return pattern
    
    regex = match.group(1).replace(r'\/', '/')
    flags = sum(re.RegexFlag[f.upper()] for f in match.group(2))
    return pattern, re.compile(regex, flags)


def _print(*args, **kwargs):
    """Shortcut for print(..., file=sys.stdout, flush=True)"""
    kwargs.update({'file': sys.stdout, 'flush': True})
    print(*args, **kwargs)


def _error(*args, **kwargs):
    """Shortcut for print(..., file=sys.stderr, flush=True)"""
    kwargs.update({'file': sys.stderr, 'flush': True})
    print(*args, **kwargs)


def _is_iterable(it):
    return callable(getattr(it, '__iter__', False))


def _is_dict_like(it):
    return callable(getattr(it, 'keys', False)) \
        and callable(getattr(it, 'values', False))


def _serialize(expect_returned, check_value=True, _depth=MAX_DEPTH):
    if _depth <= 0:
        raise RuntimeError(f'_serialize reached maximum depth of {MAX_DEPTH}')

    if check_value:
        return expect_returned

    expect_type = type(expect_returned)
    if not _is_iterable(expect_returned):
        return expect_type
    # Recursive
    return expect_type(_serialize(er, check_value=False, _depth=_depth - 1) for er in expect_returned)


print('Assertions imported OK')


