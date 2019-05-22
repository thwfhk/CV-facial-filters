#!/bin/bash

echo "[download] dlib"
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f $DIR/shape_predictor_68_face_landmarks.dat ]; then
    wget -N --tries=2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O $DIR/shape_predictor_68_face_landmarks.dat.bz2
    echo "[download] end"

    bunzip2 -f $DIR/shape_predictor_68_face_landmarks.dat.bz2
    rm $DIR/shape_predictor_68_face_landmarks.dat.bz2
fi
