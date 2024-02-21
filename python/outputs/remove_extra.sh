#!/bin/bash


# Path to markdown
md_file="results.md"

# Move path
move_dir="./keepruns/"

# Extract out unique file numbers
unique_identifier=$(awk -F '|' '{print $2}' "${md_file}" | tail -n +3)

# iterate over files in directory
for file in ./*.csv; do
  filename=$(basename "$file")
  # Check if uniq file in filename
  for identifier in $unique_identifier; do
    if [[ $filename == *"$identifier"* ]]; then
      echo "Keeping File: $filename"
      mv "$filename" "$move_dir"
      break
    fi
  done
done

rm *.csv
