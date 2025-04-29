#!/bin/bash

# Find the PID of the process LISTENING on port 5678
pid=$(lsof -iTCP:5678 -sTCP:LISTEN -t)

# Check if a PID was found
if [ -n "$pid" ]; then
  echo "Killing process $pid listening on port 5678"
  kill "$pid"
else
  echo "No process found listening on port 5678"
fi
