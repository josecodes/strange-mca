"""Test file with intentional linting issues."""


def bad_function():  # Extra space in parentheses
    x = 1 + 2  # Missing spaces around operators

    # Too many blank lines

    print("This is a test")  # Using print (but we ignore T201)
    return x
