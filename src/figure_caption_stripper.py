import re

def strip_figure_captions(text):
    """
    Strips out figure captions and their surrounding empty lines from text that match the pattern:
    FIGURE X.X followed by text starting with a capital letter or parenthesis
    
    Args:
        text (str): Input text containing figure captions
        
    Returns:
        str: Text with figure captions and surrounding empty lines removed
    """
    patterns = [r'^\s*FIGURE\s+\d+\.\d+\s+[A-Z(].*$',   # Caption
                    r'^\s*\d{1,4}\s*$',   # Page number
                    r'^\s*Chapter\s+\d+\s+[A-Z].*$',    # Chapter 
                    r'^\s*[\d\.]+\s+[A-Z].*$',   # Section
                    ]
    lines = text.split('\n')
    
    # Remove figure captions and consolidate empty lines
    filtered_lines = []
    skip_next_empty = False
    
    
    for i, line in enumerate(lines):
        is_figure = False
        # Check if current line is a figure caption
        for pattern in patterns:
            is_figure = bool(re.match(pattern, line))
            if is_figure:
                print(pattern)
                print(f"({i}){line}\n")  
                break
        
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
