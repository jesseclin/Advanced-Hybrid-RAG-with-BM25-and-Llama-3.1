import logging
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

def pdf_extract(filepath):
    config = {
        "output_format": "json",
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer()
    )
    rendered = converter(filepath)
    text, metadata, images = text_from_rendered(rendered)
    return text, metadata, images

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    logging.basicConfig(level=logging.INFO)
    
    parser.add_argument('file', help='Input PDF File') 
    args = parser.parse_args() 

    pdf_file_path = args.file
    text, metadata, images = pdf_extract(pdf_file_path)

    print(text)
    print(images)
    print(metadata)
    