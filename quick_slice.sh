#/bin/sh
RealmNumber=$1

# python pipeline/run.py svgs/$RealmNumber.svg

wine FileToVox-v1.13-win/FileToVox.exe \
    --i output/hslices \
    -o MagicaVoxel-0.99.6.4-win64/vox/wmap_$RealmNumber \
    --p output/palette_$RealmNumber.png

python pipeline/vox_chirurgy.py $RealmNumber

# for RealmNumber in {10..50}
# do
#     python pipeline/run.py svgs/$RealmNumber.svg

#     wine FileToVox-v1.13-win/FileToVox.exe \
#         --i output/hslices \
#         -o MagicaVoxel-0.99.6.4-win64/vox/wmap_$RealmNumber \
#         --p output/palette_$RealmNumber.png

#     python pipeline/vox_chirurgy.py $RealmNumber
# done
