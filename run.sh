#!/bin/sh

function usage {
  cat <<EOM
Usage: $(basename "$0") [OPTION]...
  -h          Display help
  -i VALUE    input video file
  -d VALUE    output directory
EOM
  exit 2
}
while getopts i:d: OPT
do
  case $OPT in
    "i" ) input="$OPTARG";;
    "d" ) day="$OPTARG";;
    "-h"|"--help"|* ) usage
  esac
done

python trim.py --input ./input/$input \
                --output ./output/$output \
