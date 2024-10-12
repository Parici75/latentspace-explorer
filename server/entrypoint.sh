#!/bin/bash

# Forwards SIGTERM to server to gracefully kill it
# Standard behavior is to ignore signal while process is running
_term() {
  echo "Caught SIGTERM signal!"
  echo "killing $child"
  kill -TERM "$child" 2>/dev/null
  wait "$child"
}

# Catching SIGINT will ensure that we have the same behavior in terminal (Ctrl-C)
# and in docker SIGTERM when stopping the server.
trap _term SIGTERM
trap _term SIGINT

echo "Running Dash App on host $HOSTNAME:$PORT... with '$@'"
gunicorn server.app:server --bind "$HOSTNAME:$PORT" --workers=$WORKERS --access-logfile '-' --timeout 600 $@ &

child=$!
wait "$child"
