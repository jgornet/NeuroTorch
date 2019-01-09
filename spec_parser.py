#!/usr/bin/env python

import re
import json
from fnmatch import fnmatch
import os.path
import os
import argparse


def spec_parser(filename, directory):
    volume_list = os.listdir(directory)
    volume_list = filter(lambda f: fnmatch(f, '*.tif'), volume_list)
    volume_list = list(map(lambda f: os.path.join(os.path.abspath(directory), f),
                           volume_list))

    spec = []
    for volume in volume_list:
        coord = re.findall('\d+', os.path.basename(volume))
        coord = list(map(int, coord))
        coord = [(coord[0]-1) * 2048, (coord[1]-1) * 1024, coord[2], coord[3]]
        edges = ([coord[0], coord[1], coord[2]],
                 [coord[0] + 2048, coord[1] + 1024, coord[3]+1])

        volume_spec = dict()
        volume_spec["filename"] = volume
        volume_spec["bounding_box"] = edges

        spec.append(volume_spec)

    with open(filename, 'w') as f:
        spec.sort(key=lambda volume: volume["filename"])
        f.write(json.dumps(spec, indent=2))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generates a specification' +
                                     'from a directory of TIFF files')
    parser.add_argument('DIRECTORY', help='Directory of TIFF files')
    parser.add_argument('OUTPUT', help='Output path')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    spec_parser(args.OUTPUT, args.DIRECTORY)


if __name__ == '__main__':
    main()
