# CHILDLIKE VERSION
# ======================
mkdir subset
cp *0.png subset
cd subset
mkdir crop
mogrify -path crop -crop 250x600+500+0 +repage *.png      # w x h + offset_x + offset_y
cd crop

mkdir subset2   # choose subset of frames
for file in 0000{140..410..10}.png; do cp $file subset2; done
cd subset2

mkdir alpha05
mogrify -path alpha05 -alpha set -background none -channel A -evaluate multiply 0.5 +channel *.png
cd alpha05
for file in 0000{140..410..20}.png; do cp ../$file .; done

montage *.png -tile x1 -geometry -30+0 out.png




# PROUD WAVING VERSION
# ======================
mkdir subset
cp *0.png subset
cd subset
mkdir crop
mogrify -path crop -crop 420x600+450+0 +repage *.png    # > CHANGED for proud waving
cd crop

mkdir subset2   # choose subset of frames
for file in 0000{140..410..10}.png; do cp $file subset2; done
cd subset2

mkdir alpha05
mogrify -path alpha05 -alpha set -background none -channel A -evaluate multiply 0.5 +channel *.png
cd alpha05
for file in 0000{140..410..20}.png; do cp ../$file .; done

convert -size 265x600 canvas:transparent 0.png   # > CHANGED for proud waving
convert -size 350x600 canvas:transparent Z.png   # > CHANGED for proud waving
montage *.png -tile x1 -geometry -115+0 out.png  # > CHANGED for proud waving