"""This script works around the limitation of `myst_nb` to render attachment in notebook cell inputs.
It converts every attachment in the original jupyter notebooks found in `notebooks_path` to bytes,
and insert them inplace.

This script is to be run before the `sphinx-build` command.

Inspired from https://github.com/jupyter/nbconvert/issues/699#issuecomment-372441219.
"""

import glob
import logging
import os
import sys

import nbconvert
import nbformat

logger = logging.getLogger()


def get_script_path() -> str:
    return os.path.dirname(os.path.realpath(sys.argv[0]))


notebooks_references = glob.glob(os.path.join(get_script_path(), "notebooks", "*.ipynb"))

for notebook_ref in notebooks_references:

    logger.info(f"Converting attachments of {notebook_ref}...")
    with open(notebook_ref) as nb_file:
        nb_contents = nb_file.read()

    # Convert using the ordinary exporter
    notebook = nbformat.reads(nb_contents, as_version=4)
    exporter = nbconvert.NotebookExporter()
    body, res = exporter.from_notebook_node(notebook)

    # Create a dict mapping all image attachments to their base64 representations
    images = {}
    for cell in notebook["cells"]:
        if "attachments" in cell:
            attachments = cell["attachments"]
            for filename, attachment in attachments.items():
                for mime, base64 in attachment.items():
                    images[f"attachment:{filename}"] = f"data:{mime};base64,{base64}"

    # Fix up the notebook and write it to disk
    for src, base64_coding in images.items():
        body = body.replace(f"{src}", f"{base64_coding}")
    with open(
        notebook_ref,
        "w",
    ) as output_file:
        output_file.write(body)
