from termcolor import colored


def raise_exception(e, err_str):
    """
    Raises an exception with colored error messages with explicit reference
    to the package 'aiutils'.

    Usage:
    try:
        <some code>
    except Exception as e:
        raise_exception(e, '<error message>')
    """
    pkg_err_str = 'Exception raised in package aiutils: \n{}'.format(err_str)
    print colored(pkg_err_str, 'red')
    raise


def check_assertion(cond, err_str):
    """
    Throws a colored error message if assertion fails with explicit reference
    to the package 'aiutils'.

    Usage:
    check_assertion(3==5, '<error message>')
    """
    pkg_err_str = '\nAssertion failed in package aiutils: \n{}'.format(err_str)
    assert (cond), colored(pkg_err_str, 'blue')
