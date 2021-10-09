#!/bin/sh

if [ "$#" -ne 2 ] || ! [ -f "$1" ]; then
  echo "Usage: $0 <dataset_file> <output_folder>" >&2
  exit 1
fi

if [ -d "$2" ]; then
  echo "Output folder must not exists" >&2
  exit 1
fi

DATSET_FILE=$1
OUTPUT_FOLDER=$2
MODEL=./models/model.pth
PCA_MODEL=./models/pca.pkl
FEATURES=./assets/features/train-features.json
DATASET_FOLDER=./datasets/lesions

mkdir "$OUTPUT_FOLDER"

jq -r ".images | .[] | .file_name" "$DATSET_FILE" |
while read -r file_name
do
  echo "Processing $file_name"
  file=$DATASET_FOLDER/images/$file_name
  mkdir "$OUTPUT_FOLDER/$file_name"
  cp "$file" "$OUTPUT_FOLDER/$file_name/input.jpg"
  python predict-file.py --model "$MODEL" --file "$file" --output "$OUTPUT_FOLDER/$file_name/prediction.jpg"
  ./explain.sh --model "$MODEL" --pca-model "$PCA_MODEL" --features-database "$FEATURES" --dataset-folder "$DATASET_FOLDER" --file "$file"
  mv report.pdf "$OUTPUT_FOLDER/$file_name/report.pdf"
done