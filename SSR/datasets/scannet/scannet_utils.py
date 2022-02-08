import json
import os
import numpy as np
import csv

def load_scannet_label_mapping(path):
    """ Returns a dict mapping scannet category label strings to scannet Ids

    scene****_**.aggregation.json contains the category labels as strings 
    so this maps the strings to the integer scannet Id

    Args:
        path: Path to the original scannet data.
              This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from strings to ints
            example:
                {'wall': 1,
                 'chair: 2,
                 'books': 22}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, name = int(line[0]), line[1]
            mapping[name] = scannet_id

    return mapping


def load_scannet_nyu40_mapping(path):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                 2: 5,
                 22: 23}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id = int(line[0]), int(line[4])
            mapping[scannet_id] = nyu40id
    return mapping


def load_scannet_nyu13_mapping(path):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                 2: 5,
                 22: 23}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id = int(line[0]), int(line[5])
            mapping[scannet_id] = nyu40id
    return mapping