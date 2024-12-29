import re
import unittest

def strip_figure_captions(text):
    """
    Strips out figure captions and their surrounding empty lines from text that match the pattern:
    FIGURE X.X followed by text starting with a capital letter or parenthesis
    
    Args:
        text (str): Input text containing figure captions
        
    Returns:
        str: Text with figure captions and surrounding empty lines removed
    """
    pattern = r'^\s*FIGURE\s+\d+\.\d+\s+[A-Z(].*$'
    lines = text.split('\n')
    
    # Remove figure captions and consolidate empty lines
    filtered_lines = []
    skip_next_empty = False
    
    for i, line in enumerate(lines):
        # Check if current line is a figure caption
        is_figure = bool(re.match(pattern, line))
        
        # Check if current line is empty
        is_empty = not line.strip()
        
        # Skip figure captions and manage empty lines
        if is_figure:
            skip_next_empty = True
            continue
        
        # Skip empty line after figure caption
        if is_empty and skip_next_empty:
            skip_next_empty = False
            continue
            
        filtered_lines.append(line)
    
    # Remove consecutive empty lines
    result = []
    prev_empty = False
    for line in filtered_lines:
        is_empty = not line.strip()
        if not (is_empty and prev_empty):
            result.append(line)
        prev_empty = is_empty
            
    return '\n'.join(result)

class TestFigureCaptionStripper(unittest.TestCase):
    def test_basic_figure_removal(self):
        input_text = """This is a regular paragraph.
FIGURE 1.1 Basic figure caption
Another regular paragraph."""
        expected = """This is a regular paragraph.
Another regular paragraph."""
        self.assertEqual(strip_figure_captions(input_text), expected)

    def test_multiple_figures(self):
        input_text = """Start text.
FIGURE 1.1 First caption
Middle text.
FIGURE 2.3 Second caption
End text."""
        expected = """Start text.
Middle text.
End text."""
        self.assertEqual(strip_figure_captions(input_text), expected)

    def test_parentheses_caption(self):
        input_text = """Text here.
FIGURE 1.1 (This caption starts with parentheses)
More text."""
        expected = """Text here.
More text."""
        self.assertEqual(strip_figure_captions(input_text), expected)

    def test_leading_spaces(self):
        input_text = """Text here.
    FIGURE 1.1 Caption with leading spaces
More text."""
        expected = """Text here.
More text."""
        self.assertEqual(strip_figure_captions(input_text), expected)

    def test_no_figures(self):
        input_text = """Just regular text.
No figures here.
More regular text."""
        self.assertEqual(strip_figure_captions(input_text), input_text)

    def test_edge_cases(self):
        test_cases = [
            # Empty string
            ("", ""),
            # Single line with figure
            ("FIGURE 1.1 Single line", ""),
            # Figure-like but not matching pattern
            ("FIGURES 1.1 Wrong format\nFIGURE 1.1a Invalid number", "FIGURES 1.1 Wrong format\nFIGURE 1.1a Invalid number"),
            # Multiple consecutive figures
            ("FIGURE 1.1 First\nFIGURE 1.2 Second", ""),
        ]
        for input_text, expected in test_cases:
            self.assertEqual(strip_figure_captions(input_text), expected)

    def test_real_world_example(self):
        input_text = """Introduction to Data Structures

FIGURE 1.1 Binary Search Tree Implementation

The binary search tree is a fundamental data structure that provides
efficient searching capabilities.

FIGURE 1.2 (Example of tree traversal methods)

As shown above, tree traversal can be implemented in multiple ways.
"""
        expected = """Introduction to Data Structures

The binary search tree is a fundamental data structure that provides
efficient searching capabilities.

As shown above, tree traversal can be implemented in multiple ways.
"""
        self.assertEqual(strip_figure_captions(input_text), expected)

if __name__ == '__main__':
    unittest.main()
