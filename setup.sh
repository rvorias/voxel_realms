#/bin/sh

# download FileToVox
for file in FileToVox-v1.13-win FileToVox-v1.13-linux; do
    rm -rf ./$file
    wget https://github.com/Zarbuz/FileToVox/releases/download/1.13/$file.zip
    unzip $file.zip -d ./$file && rm $file.zip
done 

# download MagicaVoxel
file=MagicaVoxel-0.99.6.4-win64
wget https://github.com/ephtracy/ephtracy.github.io/releases/download/0.99.6/$file.zip
unzip $file.zip -d ./ && rm $file.zip