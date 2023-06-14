import io
import pickle
import re
import sys
import types

from contextlib import redirect_stdout
from copy import deepcopy
from itertools import chain, zip_longest

# Constants
from pprint import pprint

student_user = True
ASSERTION_FILE = 'assertions.pickle'
MAX_DEPTH = 20
ARGUMENTS = 'arguments'
EXPECT_PRINTED = 'expect_printed'
EXPECT_RETURNED = 'expect_returned'
EXPECT_RAISED = 'expect_raised'
EXPECT_INPUTS = 'expect_inputs'
CHECK_VALUE = 'check_value'



def create_assertion(reference, expectations, assertion_file=ASSERTION_FILE, _debug=True, _lazy=False):
    """Run teacher's function, check expectations, and store in assertions.pickle"""

    # Validate function arguments

    assert type(reference) is types.FunctionType, \
        f'Teacher bug; reference must be a function, got {type(reference)}'

    assert isinstance(expectations, (list, tuple)), \
        f'Teacher bug; expectations must be a list or tuple, got {type(expectations)}'

    assert type(_debug) is bool, \
        f'Teacher bug; _debug must be a boolean, got {type(_debug)}'

    assert type(_lazy) is bool, \
        f'Teacher bug; _lazy must be a boolean, got {type(_lazy)}'

    label = reference.__name__

    save_expectations = list()

    for expectation in expectations:

        # Validate whether expectation is dictionary
        assert isinstance(expectation, dict), \
            f'Teacher bug; expectation should be dictionary, not {type(expectation)}'

        # Validate arguments, coerce it to tuple if necessary
        arguments = expectation.get(ARGUMENTS)
        if arguments is None:
            arguments = ()
        else:
            try:
                arguments = tuple(arguments)
            except TypeError:
                # Not an iterable, make it so
                arguments = (arguments,)

        # Validate expected inputs
        expect_inputs = expectation.get(EXPECT_INPUTS)
        if expect_inputs is not None:
            if isinstance(expect_inputs, str):
                expect_inputs = (expect_inputs,)
            else:
                assert type(expect_inputs) in {list, tuple}, \
                f'Teacher bug; inputs must be list or tuple, got "{expect_inputs}"'
                expect_inputs = tuple(str(i) for i in expect_inputs)

        # In lazy mode, we assume that the reference implementation is correct,
        # and we use it to cheat to obtain expected printed, returned and raised.
        # Otherwise, they have to be provided explicitly, meaning that we will
        # effectively also test the reference implementation for correctness.

        if _lazy:
            print(f'Warning, your reference implementation will NOT be tested!')
            cheating = _call_function(reference, arguments, expect_inputs)
            expect_printed, expect_returned, expect_raised = cheating
        else:
            # Validate expected printed lines, cleaned up and stripped
            expect_printed = _clean_output(expectation.get(EXPECT_PRINTED))

            # Validate expected return value
            expect_returned = expectation.get(EXPECT_RETURNED)

            # Validate expected exception type, str, or (type, str)
            expect_raised = expectation.get(EXPECT_RAISED)
            if type(expect_raised) is str:
                expect_raised = None, expect_raised
            elif type(expect_raised) is type:
                expect_raised = expect_raised, None
            elif isinstance(expect_raised, BaseException):
                expect_raised = type(expect_raised), str(expect_raised)
            elif expect_raised is not None:
                assert (type(expect_raised) is tuple \
                and tuple(type(r) for r in expect_raised) == (type, str)), \
                f'Teacher bug; exception must be type, str or (type, str), got "{expect_raised}"'


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

        expectation = {
        }
        if arguments:
            expectation[ARGUMENTS] = arguments
        if expect_inputs:
            expectation[EXPECT_INPUTS] = expect_inputs
        if expect_printed:
            expectation[EXPECT_PRINTED] = expect_printed
        if expect_returned:
            if not check_value:
                expect_returned = type(expect_returned)
            expectation[EXPECT_RETURNED] = expect_returned
        if expect_raised:
            expectation[EXPECT_RAISED] = expect_raised

        if not check_value:
            expect_returned[CHECK_VALUE] = False

        save_expectations.append(expectation)

    # Teacher sanity, visually check the generated assertions
    if _debug:
        pprint(save_expectations)

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


def check_assertion(assignment, label=None, fail_fast=True, assertion_file=ASSERTION_FILE):
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
        arguments = deepcopy(expectation.get(ARGUMENTS))
        expect_printed = deepcopy(expectation.get(EXPECT_PRINTED))
        expect_returned = deepcopy(expectation.get(EXPECT_RETURNED))
        expect_raised = deepcopy(expectation.get(EXPECT_RAISED))
        expect_inputs = deepcopy(expectation.get(EXPECT_INPUTS))
        check_value = deepcopy(expectation.get(CHECK_VALUE, True))

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
        if not check_value:
            returned = _get_type(returned)

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

        # Check printed lines, if applicable (expect_printed should not be empty string)
        if expect_printed != '':
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

def _call_function(function, arguments=None, inputs=None):
    """Call function with given arguments, capture output, return value or exception"""

    returned = None
    raised = None

    if not arguments:
        arguments = tuple()

    orig_input = None
    if inputs is not None:
        input_index = 0

        def _alt_input(*args, **kwargs):
            nonlocal input_index
            if input_index < len(inputs):
                input_index += 1
                input = inputs[input_index - 1]
                if not isinstance(input, str):
                    input = str(input)
                return input
            return ''

        orig_input = __builtins__['input']
        __builtins__['input'] = _alt_input

    with io.StringIO() as buffer, redirect_stdout(buffer):
        try:
            returned = function(*arguments)
        except Exception as exc:
            raised = type(exc), str(exc)
        printed = _clean_output(buffer.getvalue())

    if orig_input is not None:
        __builtins__['input'] = orig_input

    return printed, returned, raised


def _clean_output(printed):
    """Splits the output in lines, trims each of them, and filters out empty ones."""

    if printed is None:
        return None

    if type(printed) is str:
        printed = printed.split('\n')
    elif _is_iterable(printed):
        printed = (_clean_output(p) for p in printed)
        printed = chain(*(p for p in printed if p is not None))
    else:
        assert False, f'Teacher bug; printed output must be None, str, or iterable of those'

    if printed:
        printed = (line.strip() for line in printed if line is not None)
        printed = (line for line in printed if len(line) > 0 and line[0] != '#')
        printed = tuple(printed)
    return printed or None


def _check_printed(expect, test, *, subject='Printed output'):
    """Check if the printed lines matches the expectation strings and/or regexes"""

    errors = []

    if expect is None:
        if test is not None:
            errors.append(f'{subject} was unexpected')
            _error(errors[0])
            return '\n'.join(errors)
        return None

    elif test is None:
        errors.append(f'{subject} was expected, but got nothing')
        _error(errors[0])
        return '\n'.join(errors)

    for pattern, line in zip_longest(expect, test):
        if pattern is None:
            errors.append(f'{subject} "{line}" was unexpected')
            continue
        if line is None:
            errors.append(f'{subject} "{pattern}" was expected')
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

    if expect is None:
        if test is not None:
            return f'{subject} was {type(test).__name__} {test!r} but None was expected'
        return
    elif test is None:
        return f'{subject} was None but {type(expect).__name__} {expect!r} was expected'

    expect_type = type(expect)
    if not isinstance(test, expect_type):
        return f'{subject} was {type(test).__name__} {test!r}' \
               + f' but a value of type {expect_type.__name__} was expected'

    if check_value and expect_type is str:
        return _check_pattern(expect, test, subject=subject)

    if check_value and expect_type is float:
        if f'{expect:.6f}' != f'{test:.6f}':
            return f'{subject} {type(test).__name__} {test!r} is unequal' \
                   + f' to expected {expect_type.__name__} {expect!r}'
        else:
            return None

    if _is_iterable(expect):
        if not _is_iterable(test):
            return f'{subject} {type(test).__name__} {test!r} expected to be iterable'
        
        if _is_dict_like(expect):
            if not _is_dict_like(test):
                return f'{subject} {type(test).__name__} {test!r} expected to be dict-like'
            indices = sorted(expect.keys())
            if indices != sorted(test.keys()):
                return f'{subject} {type(test).__name__} {test!r} expected to contain key(s) ' \
                       + ', '.join(f'{idx!r}' for idx in indices)
            if check_value:
                if test == expect:
                    return None
                return f'{subject} {type(test).__name__} expected to contain' \
                     f' {expect!r}, but got {test!r}'
            else:
                for i in indices:
                    expect_item = expect[i]
                    test_item = test[i]
                    msg = _check_returned(expect_item, test_item, subject='', check_value=check_value, _depth=_depth - 1)
                    if msg:
                        return f'{subject} {type(test).__name__} {test!r} at key {i!r}:{msg}'
                return None

        else:
            indices = list(range(len(expect)))
            expect_list = list(expect)
            test_list = list(test)
        
        expect_len = len(expect_list)
        test_len = len(test_list)
        if expect_len != test_len:
            return f'{subject} {type(test).__name__} {test!r} expected to contain' \
                 f' {expect_len} items, but got {test_len}'

        for i, expect_item, test_item in zip(indices, expect_list, test_list):
            msg = _check_returned(expect_item, test_item, subject='', check_value=check_value, _depth=_depth - 1)
            if msg:
                return f'{subject} {type(test).__name__} {test!r} at index {i!r}:{msg}'

    elif check_value and expect != test:
        return f'{subject} {type(test).__name__} {test!r} is unequal' \
               + f' to expected {expect_type.__name__} {expect!r}'


def _check_raised(expect, test, *, subject='Exception'):
    """Check that the raised exception, if any, matches the expected exception"""

    if expect is not None and isinstance(expect, BaseException):
        expect = type(expect), str(expect)

    if test is not None and isinstance(test, BaseException):
        test = type(test), str(test)

    if expect is None and test is not None:
        raised_type, raised_str = test
        return f'{subject} {raised_type.__name__} was raised but not expected: {raised_str}'
    
    elif expect is not None and test is None:
        expect_type, expect_pattern = expect
        return f'{subject} {expect_type.__name__} was expected but not raised'
    
    elif test is not None:  # and expect is not None
        expect_type, expect_pattern = expect
        raised_type, raised_str = test
        
        error = None
        if expect_type is not None:
            if expect_type != raised_type:
                error = f'{subject} {raised_type.__name__} was raised,' \
                        + f' but {expect_type.__name__} was expected'

        if error is None and expect_pattern is not None:
            error = _check_pattern(expect_pattern, raised_str, subject=subject)

        return error


def _check_pattern(expect, test, subject=''):
    """Check if given value matches the expectation string or regex"""

    if expect is None:
        if test is None:
            return None
        return f'{subject} "{test}" was unexpected'
    else:
        expect, regex = _make_pattern(expect)
        if regex is None:
            if test is None:
                return f'{subject} expected a string matching "{expect}", but got None'
            if expect != test:
                return f'{subject} "{test}" doesn\'t match expected string "{expect}"'
        else:
            if test is None:
                return f'{subject} expected to match the regular expression "{expect}", but got None'
            if regex.match(test) is None:
                return f'{subject} "{test}" doesn\'t match the regular expression "{expect}"'


# Regular expression to match regular expressions
_meta_rex = re.compile(r'^/(.*)(?<!\\)/([ais]*)$')


def _make_pattern(pattern):
    """If the argument matches the meta regex, compile it"""

    match = _meta_rex.match(pattern)
    if match is None:
        # Not a regex
        return pattern, None
    
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


def _get_type(returned, _depth=MAX_DEPTH):
    if _depth <= 0:
        raise RuntimeError(f'_serialize reached maximum depth of {MAX_DEPTH}')

    expect_type = type(returned)

    # Simple type
    if isinstance(returned, str) or not _is_iterable(returned):
        return expect_type

    # Recursive
    if _is_dict_like(returned):
        return expect_type(zip(
            returned.keys(),
            (_get_type(v, _depth=_depth - 1)
             for v in returned.values())
        ))

    return expect_type(
        _get_type(er, _depth=_depth - 1)
        for er in returned
    )


print('Assertions imported OK')
