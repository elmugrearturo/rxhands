import os
import argparse

from rxhands.entrypoints import symbolic, neural

parser = argparse.ArgumentParser(prog="python rxhands",
                                 description="Labels finger joints in hand x-rays",
                                 epilog="Just keep swimming")

# arguments
parser.add_argument("INPUT_FOLDER", type=str,
                    help='location of the input images')
parser.add_argument("-al", "--algorithm", choices=["symbolic", "neural"],
                    required=True,
                    help="choose the labeling algorithm: symbolic or neural")
parser.add_argument("-ch", "--crop_hands", action="store_true",
                    help='crop hands before labeling')
parser.add_argument("-of", "--output-folder", type=str,
                   help='location of the output images')

args = parser.parse_args()
input_folder = os.path.abspath(args.INPUT_FOLDER)
if args.output_folder == None:
    output_folder = os.path.join(os.path.dirname(input_folder), "results")
else:
    output_folder = os.path.abspath(args.output_folder)

if args.algorithm == "symbolic":
    print("\nLoading symbolic model:")
    print(f"\t Input folder: {input_folder}")
    print(f"\t Output folder: {output_folder}")
    if args.crop_hands:
        print("\t Symbolic model doesn't support hand cropping.")
    print(f"\t Crop hands before labeling?: False\n")
    
    symbolic(data_folder=input_folder,
             results_folder=output_folder)
else:
    print("\nLoading neural model:")
    print(f"\t Input folder: {input_folder}")
    print(f"\t Output folder: {output_folder}")
    print(f"\t Crop hands before labeling?: {args.crop_hands}\n")