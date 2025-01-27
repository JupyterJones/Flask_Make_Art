#!/bin/bash
# filename: vidall.sh

function vidall() {
    # Ensure a directory path is provided
    if [ -z "$1" ]; then
        echo "Usage: vidall <directory_path>"
        exit 1
    fi

    # Check if the input directory exists
    input_dir="$1"
    if [ ! -d "$input_dir" ]; then
        echo "Directory $input_dir does not exist."
        exit 1
    fi

    # Get the base name of the input directory
    dir_name=$(basename "$input_dir")

    # Output directory and video file name
    output_dir="static/videos"
    output_file="$output_dir/$dir_name.mp4"

    # Ensure the output directory exists
    mkdir -p "$output_dir"

    # Check for image files in the input directory
    if ls "$input_dir"/*.png 1> /dev/null 2>&1 || ls "$input_dir"/*.jpg 1> /dev/null 2>&1; then
        echo "Creating video from images in directory: $input_dir"

        # Create a temporary file list
        filelist="filelist.txt"
        > $filelist  # Create or empty the filelist

        for f in "$input_dir"/*.png "$input_dir"/*.jpg; do
            # Ensure the file exists before adding it to the list
            if [ -f "$f" ]; then
                echo "file '$f'" >> $filelist
            fi
        done

        # Run ffmpeg to create the video
        ffmpeg -r 17 -f concat -safe 0 -i "$filelist" -c:v libx265 -pix_fmt yuv420p -y "$output_file"

        # Clean up the temporary file list
        rm -f $filelist
        echo "Video created successfully: $output_file"
    else
        echo "No .png or .jpg files found in the directory $input_dir."
    fi
}
