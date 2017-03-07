#!/bin/bash
mkdir -p _ignore

curl -O https://notendur.hi.is/~pmelsted/kennsla/rei/albums.zip
curl -O https://notendur.hi.is/~pmelsted/kennsla/rei/mnist.zip

unzip albums.zip
unzip mnist.zip

rm "albums.zip"
rm "mnist.zip"