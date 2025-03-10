#!/usr/bin/env python
"""
Test script for the parse_strange_loop_response function.
"""

import sys
from src.strange_mca.graph import parse_strange_loop_response

def test_parse_response():
    """Test the parse_strange_loop_response function with different formats."""
    
    # Test case 1: New format with asterisks before and after
    test_response_1 = """
Here's my analysis of your response:

The response is good but could be improved by adding more specific examples.

Final Response:
**************************************************
This is the final response content.
It spans multiple lines.
**************************************************

I hope this helps!
"""

    # Test case 2: Old format with Final Response: directly followed by content
    test_response_2 = """
Here's my analysis:

Final Response:
This is the final response in the old format.
It also spans multiple lines.
**************************************************

End of response.
"""

    # Test case 3: Mixed format with equals signs
    test_response_3 = """
Analysis complete.

Final Response:
================================================================================
This is the final response with equals signs.
Multiple lines here too.
================================================================================

Done.
"""

    # Test case 4: No clear format
    test_response_4 = """
This response doesn't have a clear final response section.
It's just a regular response.
"""

    # Test case 5: Empty response
    test_response_5 = ""

    # Run tests
    print("Test Case 1 (New Format):")
    print("-" * 40)
    result_1 = parse_strange_loop_response(test_response_1)
    print(f"Result: {result_1}")
    print()

    print("Test Case 2 (Old Format):")
    print("-" * 40)
    result_2 = parse_strange_loop_response(test_response_2)
    print(f"Result: {result_2}")
    print()

    print("Test Case 3 (Equals Signs):")
    print("-" * 40)
    result_3 = parse_strange_loop_response(test_response_3)
    print(f"Result: {result_3}")
    print()

    print("Test Case 4 (No Format):")
    print("-" * 40)
    result_4 = parse_strange_loop_response(test_response_4)
    print(f"Result: {result_4}")
    print()

    print("Test Case 5 (Empty):")
    print("-" * 40)
    result_5 = parse_strange_loop_response(test_response_5)
    print(f"Result: {result_5}")
    print()

if __name__ == "__main__":
    test_parse_response() 