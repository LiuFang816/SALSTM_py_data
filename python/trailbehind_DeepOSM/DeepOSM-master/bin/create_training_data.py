#!/usr/bin/env python

"""Create training data from OpenStreetMap labels and NAIP images."""

import argparse
from src.training_data import download_and_serialize


def create_parser():
    """Create the argparse parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size",
                        default=64,
                        type=int,
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--tile-overlap",
                        default=1,
                        type=int,
                        help="divide the tile-size by this arg for how many pixels to move over "
                             "when tiling data. this is set to 1 by default, so tiles don't "
                             "overlap. setting it to 2 would make tiles overlap by half, and "
                             "setting it to 3 would make the tiles overlap by 2/3rds")
    parser.add_argument("--pixels-to-fatten-roads",
                        default=3,
                        type=int,
                        help="the number of px to fatten a road centerline (e.g. the default 3 "
                             "makes roads 7px wide)")
    parser.add_argument("--percent-for-training-data",
                        default=.90,
                        type=float,
                        help="how much data to allocate for training. the remainder is left for "
                             "test")
    parser.add_argument("--bands",
                        default=[1, 1, 1, 1],
                        nargs=4,
                        type=int,
                        help="specify which bands to activate (R  G  B  IR)"
                             "--bands 1 1 1 1 (which activates only all bands)")
    parser.add_argument(
        "--label-data-files",
        nargs='+',
        default=[
            'http://download.geofabrik.de/north-america/us/delaware-latest.osm.pbf',
        ],
        type=str,
        help="PBF files to extract road/feature label info from")
    parser.add_argument("--naip-path",
                        default=['de', '2013'],
                        nargs=2,
                        type=str,
                        help="specify the state and year for the NAIPs to analyze"
                             "--naip-path de 2013 (defaults to some Delaware data)")
    parser.add_argument("--randomize-naips",
                        default=False,
                        action='store_false',
                        help="turn on this arg if you don't want to get NAIPs in order from the "
                             "bucket path")
    parser.add_argument("--number-of-naips",
                        default=6,
                        type=int,
                        help="the number of naip images to analyze, 30+ sq. km each")
    parser.add_argument("--extract-type",
                        default='highway',
                        choices=['highway', 'tennis', 'footway', 'cycleway'],
                        help="the type of feature to identify")
    parser.add_argument("--save-clippings",
                        action='store_true',
                        help="save the training data tiles to /data/naip")
    return parser


def main():
    """Download and serialize training data."""
    args = create_parser().parse_args()
    naip_state, naip_year = args.naip_path
    download_and_serialize(args.number_of_naips,
                           args.randomize_naips,
                           naip_state,
                           naip_year,
                           args.extract_type,
                           args.bands,
                           args.tile_size,
                           args.pixels_to_fatten_roads,
                           args.label_data_files,
                           args.tile_overlap)


if __name__ == "__main__":
    main()
