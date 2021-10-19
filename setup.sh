#/bin/sh

file=FileToVox-v1.13-win

rm -rf ./$file
mkdir ./$file
https://github.com/Zarbuz/FileToVox/releases/download/1.13/
wget https://github.com/Zarbuz/FileToVox/releases/download/1.13/$file.zip
unzip $file -d ./$file.zip && rm $file