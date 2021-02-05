#!/usr/bin/env bash

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/zip -O tlt_trafficcamnet_pruned_v1.0.zip && \
unzip tlt_trafficcamnet_pruned_v1.0.zip