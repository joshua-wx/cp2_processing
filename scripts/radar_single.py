"""
Raw radar PPIs processing. Quality control, filtering, attenuation correction,
dealiasing, unfolding, hydrometeors calculation, rainfall rate estimation.
Tested on CP2.

@title: cp2_processing
@author: Valentin Louf <valentin.louf@bom.gov.au> and Joshua Soderholm <joshua.soderholm@bom.gov.au>
@institution: Bureau of Meteorology
@date: 29/04/2020
@version: 1

.. autosummary::
    :toctree: generated/

    main
"""
# Python Standard Library
import os
import argparse
import warnings

# Other Libraries
import crayons


def main():
    """
    Just print a welcoming message and calls the production_line_multiproc.
    """
    # Start with a welcome message.
    print("#" * 79)
    print("")
    print(" " * 25 + crayons.red("Raw radar PPIs production line.\n", bold=True))
    print(" - Input data directory path is: " + crayons.yellow(INFILE))
    print(" - Output data directory path is: " + crayons.yellow(OUTPATH))
    if USE_UNRAVEL:
        print(" - " + crayons.yellow("UNRAVEL") + " will be used as dealiasing algorithm.")
    else:
        print(" - " + crayons.yellow("REGION-BASED") + " will be used as dealiasing algorithm.")
    print("\n" + "#" * 79 + "\n")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import cp2_processing
        cp2_processing.process_and_save(INFILE, OUTPATH, use_unravel=USE_UNRAVEL)

    print(crayons.green("Process completed."))

    return None


if __name__ == '__main__':
    """
    Global variables definition and logging file initialisation.
    """
    # Parse arguments
    parser_description = """Raw radar PPIs processing. It provides Quality
control, filtering, attenuation correction, dealiasing, unfolding, hydrometeors
calculation, and rainfall rate estimation."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-i',
        '--input',
        dest='infile',
        type=str,
        help='Input file',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        dest='outdir',
        type=str,
        help='Output directory.',
        required=True)

    parser.add_argument('--unravel', dest='unravel', action='store_true')
    parser.add_argument('--no-unravel', dest='unravel', action='store_false')
    parser.set_defaults(unravel=True)

    args = parser.parse_args()
    INFILE = args.infile
    OUTPATH = args.outdir
    USE_UNRAVEL = args.unravel

    if not os.path.isfile(INFILE):
        parser.error("Invalid input file.")

    if not os.path.isdir(OUTPATH):
        parser.error("Invalid (or don't exist) output directory.")

    main()

    
    # %run radar_single -i /g/data/hj10/cp2/level_1/s_band/sur/2014/20141127/cp2-s_20141127_063502.sur.mdv -o /scratch/kl02/jss548/cp2_level_1b/v2020/