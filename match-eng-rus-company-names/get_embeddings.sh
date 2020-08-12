#!/bin/sh

# Downlad embedding files if they don't exist

echo "*** Looking for MUSE embeddings..."

# Create embeddings directiry if not exist
mkdir -p muse_embeddings

echo "*** Downloading English MUSE embeddings"
wget -P ./muse_embeddings -nc -c  https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec

echo "*** Downloading Russian MUSE  embeddings"
wget -P ./muse_embeddings -nc -c https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.ru.vec

echo "*** Done ***"


