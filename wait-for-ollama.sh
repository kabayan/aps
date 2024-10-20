#!/bin/sh
set -e

until curl -s http://ollama:11434/api/version > /dev/null; do
  echo 'Waiting for Ollama service...'
  sleep 1
done

echo "Ollama is up - pulling the model"
curl -X POST http://ollama:11434/api/pull -d '{"name": "lucas2024/gemma-2-2b-jpn-it:q8_0"}'
echo "Model pulled - executing command"