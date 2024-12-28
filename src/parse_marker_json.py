import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import argparse
   

@dataclass
class Block:
    id: str
    block_type: str
    html: str
    polygon: List[List[float]]
    children: Optional[List['Block']]
    section_hierarchy: Optional[Dict[str, str]] = None
    images: Optional[Dict] = None

def parse_block(data: Dict) -> Block:
    """Parse a JSON block into a Block object."""
    return Block(
        id=data['id'],
        block_type=data['block_type'],
        html=data['html'],
        polygon=data['polygon'],
        children=[parse_block(child) for child in data['children']] if data['children'] else None,
        section_hierarchy=data.get('section_hierarchy'),
        images=data.get('images')
    )

def process_document(json_data: Dict) -> Block:
    """Process the entire document starting from root."""
    return [parse_block(child) for child in json_data['children']] if json_data['children'] else None

def analyze_block(blocks: Block|List[Block], depth: int = 0):
    """Analyze and print information about a block and its children."""
    if isinstance(blocks, Block):
        blocks = [blocks]
        
    for block in blocks:
        indent = "  " * depth
        #if block.block_type == 'Caption':
        #if block.block_type == 'Document':
        #if block.block_type == 'Table':
        #if block.block_type == 'SectionHeader':
        if block.block_type == 'ListItem':
            #print(f"{indent}Block ID: {block.id}")
            #print(f"{indent}Type: {block.block_type}")
            print(f"{indent}HTML Content: {block.html}")  # Truncate long HTML
            #print(f"{indent}Polygon Coordinates: {block.polygon}")
    
        #if block.section_hierarchy:
        #    print(f"{indent}Section Hierarchy: {block.section_hierarchy}")
    
        #if block.images:
        #    print(f"{indent}Images: {block.images}")
    
        if block.children:
            #print(f"{indent}Children:")
            for child in block.children:
                analyze_block(child, depth + 1)

# Example usage:
def main():
    # Sample JSON string (you would typically load this from a file)
    json_str = '''
    {
        "id": "/page/10/Page/366",
        "block_type": "Page",
        "html": "<content-ref src='/page/10/SectionHeader/0'></content-ref>...",
        "polygon": [[0.0, 0.0], [612.0, 0.0], [612.0, 792.0], [0.0, 792.0]],
        "children": [
            {
                "id": "/page/10/SectionHeader/0",
                "block_type": "SectionHeader",
                "html": "<h1>Supplementary Material</h1>",
                "polygon": [[217.845703125, 80.630859375], [374.73046875, 80.630859375],
                           [374.73046875, 107.0], [217.845703125, 107.0]],
                "children": null,
                "section_hierarchy": {"1": "/page/10/SectionHeader/1"},
                "images": {}
            }
        ]
    }
    '''
    parser = argparse.ArgumentParser()
    logging.basicConfig(level=logging.INFO)
    
    parser.add_argument('file', help='Input JSON File') 
    args = parser.parse_args() 

    json_file_path = args.file
    print(json_file_path)
    # Parse JSON string into Python dictionary
    with open(json_file_path, encoding='utf-8') as file:
        data = json.load(file)
    
    # Process the document
    document = process_document(data)
    
    # Analyze the document structure
    print("Analyzing document structure:")
    analyze_block(document)
    
    # Example of accessing specific fields
    print("\nAccessing specific fields:")
    print(f"Document ID: {document[0].id}")
    print(f"Document Type: {document[0].block_type}")
    
    # Access first child if it exists
    if document[0].children:
        first_child = document[0].children[0]
        print(f"First Child ID: {first_child.id}")
        print(f"First Child Section Hierarchy: {first_child.section_hierarchy}")

if __name__ == "__main__":
    main()