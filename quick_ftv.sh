#/bin/sh

RealmNumber=$1

# this will exit with the hm param, which is stored in $?
python pipeline/run.py svgs/$RealmNumber.svg

# wine FileToVox-v1.13-win/FileToVox.exe \
#     --i output/height_$RealmNumber.png \
#     -o MagicaVoxel-0.99.6.4-win64/vox/map_$RealmNumber \
#     --hm=$? \
#     --cm output/color_$RealmNumber.png

FileToVox-v1.13-linux/FileToVox \
    --i MagicaVoxel-0.99.6.4-win64/vox/map_$RealmNumber.vox \
    --o MagicaVoxel-0.99.6.4-win64/vox/wmap_$RealmNumber \
    --s output/flood_$RealmNumber.json

python pipeline/vox_chirurgy.py $RealmNumber
