#!/bin/sh

interpreter=python3
program=../2dimCNN.py


subconfigs_path=../cnn_configs/subconfigs/

kernel_widths=$(seq 5 3 20)
epochs=200

learning_rates="0.001 0.0001 0.00001"

batch_sizes="64 128"

fully_connected_widths="64 128 256"

training_backchannels=../../datasets/train/train_pos/data.mfcc.npy
training_frontchannels=../../datasets/train/train_neg/data.mfcc.npy

validation_backchannels=../../datasets/valid/valid_pos/data.mfcc.npy
validation_frontchannels=../../datasets/valid/valid_neg/data.mfcc.npy

# iterate the cnn configurations
for cnn_config in ../cnn_configs/*.json
do

  filename=$(basename $cnn_config)
  # iterate kernel sizes
  for kernel_width in $kernel_widths
  do

    kernel_size_query="(..|objects|select(has(\"kernel_width\"))).kernel_width |= $kernel_width"

    for fully_connected_width in $fully_connected_widths
    do
      #echo $fully_connected_width
      config_file="$filename-k$kernel_width-fc$fully_connected_width"
      subconfig_file=$subconfigs_path$config_file
      fully_connected_width_query="(..|objects|select(has(\"input\"))).input |= $fully_connected_width | (..|objects|select(has(\"output\"))).output |= $fully_connected_width  | .fully_connected.fci.input=-1 | .fully_connected.fcn.output=2"
      # generate acording config file.
      jq "$kernel_size_query | $fully_connected_width_query " $cnn_config > $subconfig_file


      for learning_rate in $learning_rates
      do
        #echo $learning_rate
        for batch_size in $batch_sizes
        do
          #echo $batch_size
          #echo "hello"
          report_file="$config_file-$learning_rate-$batch_size"
          model_name="$config_file-$learning_rate-$batch_size.cnn"
          #echo "$report_file"
          $interpreter $program -b $batch_size train --report $report_file -e $epochs -C $subconfig_file -lr $learning_rate -o $model_name\
            -s $training_backchannels $training_frontchannels\
            -v $validation_backchannels $validation_frontchannels

        done


      done

    done


  done


done