mogrify -background white -alpha remove -alpha off *.png   #thistle
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -c:a aac -b:a 384k -profile:a aac_low -vf "crop=720:480:200:0" output_child.mp4
ffmpeg -i output_sexy.mp4 -i global_sexy.mp4 -filter_complex "[0]pad=iw+10:color=black[left];[left][1]hstack=inputs=2" sexy.mp4

# normal speed. Also removed audio stuff
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 30 -vf "crop=720:480:200:0" -vf "setpts=(1/2)*PTS"


### Morphing videos

mogrify -background white -alpha remove -alpha off *.png
# mogrify -background LightSteelBlue1 -alpha remove -alpha off *.png

ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 60 -vf "crop=978:550:100:0" -vf "setpts=(1/2)*PTS" output2.mp4


ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2, setpts=(1/2)*PTS" output.mp4

# add text
ffmpeg -i output2.mp4 -vf "drawtext=enable='between(t,0,5)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Neutral': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,11,19)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Depressed': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,24,31)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Childlike': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,38,45)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Strutting': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,53,61)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Sexy': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,68,73)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Proud': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,80,87)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Old': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=enable='between(t,94,97.83)': fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Neutral': \
fontsize=36: box=1: boxcolor=black@0.3: boxborderw=5: x=(w-text_w)/2: y=500, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='(a)': fontsize=48: x=200: y=30" -acodec copy output2txt.mp4



# crop final video for stitching
ffmpeg -i output2.mp4 -filter:v "crop=470:550:200:0" output2final.mp4



#####################################################################################
# Triple videos (fidelity videos: training, mtl, tl)

# For style, and {MTDS, GRU, Lagrange}, make video from Julia / Meshcat / threejs, and enter given folder.

# 1. MTDS version
mogrify -background white -alpha remove -alpha off *.png
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2, setpts=(1/2)*PTS" output.mp4

# 2. GRU version (identical)
mogrify -background white -alpha remove -alpha off *.png
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2, setpts=(1/2)*PTS" output.mp4

# 3. Lagrangian version (dark grey background)
mogrify -background darkgrey -alpha remove -alpha off *.png
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2, setpts=(1/2)*PTS" output.mp4

cp output.mp4 ../anim_{train}/{strut}_{train}_{lagr}.mp4

#########################################
# ANNOTATE

ffmpeg -i strut_train_mtds.mp4 -vf "drawbox=x=68: y=367: w=367: h=44: color=LightGrey@0.5: t=fill, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='MTDS': \
fontsize=30: box=1: boxcolor=0xE0835E@0.7: boxborderw=5: x=80: y=599, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='vs': \
fontsize=30: x=180: y=605, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Ground Truth': \
fontsize=30: box=1: boxcolor=0xCCCC00@0.7: boxborderw=5: x=220: y=599, \
drawbox=x=68: y=588: w=367: h=44: color=black@0.8" -y test.mp4

ffmpeg -i strut_train_gru.mp4 -vf "drawbox=x=20: y=588: w=454: h=44: color=LightGrey@0.5: t=fill, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='GRU-1L (closed)': \
fontsize=26: box=1: boxcolor=0x4C9DFF@0.7: boxborderw=3: x=30: y=598, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='vs': \
fontsize=26: x=245: y=605, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Ground Truth': \
fontsize=26: box=1: boxcolor=0xCCCC00@0.7: boxborderw=5: x=280: y=600, \
drawbox=x=20: y=588: w=454: h=44: color=black@0.8" -y test2.mp4

ffmpeg -i strut_train_lagr.mp4 -vf "drawbox=x=35: y=588: w=435: h=44: color=LightGrey@0.5: t=fill, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Model side-by-side comparison': \
fontsize=26: x=45: y=598, \
drawbox=x=35: y=588: w=435: h=44: color=black@0.8" -y test3.mp4


#########################################


ffmpeg -i dep_tl_lagr.mp4 -vf "drawbox=x=125: y=588: w=350: h=44: color=LightGrey@0.5: t=fill, \
drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Side-by-side comparison': \
fontsize=26: x=135: y=598, \
drawbox=x=125: y=588: w=350: h=44: color=black@0.8" -y test.mp4

"drawtext=enable='between(t,0,1)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='*-------------',
drawtext=enable='between(t,1,2)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='-*------------',
drawtext=enable='between(t,2,3)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='--*-----------',
drawtext=enable='between(t,3,4)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='---*----------',
drawtext=enable='between(t,4,5)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='----*---------',
drawtext=enable='between(t,5,6)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='-----*--------',
drawtext=enable='between(t,6,7)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='------*-------',
drawtext=enable='between(t,7,8)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='-------*------',
drawtext=enable='between(t,8,9)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='--------*-----',
drawtext=enable='between(t,9,10)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='---------*----',
drawtext=enable='between(t,10,11)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='----------*---',
drawtext=enable='between(t,11,12)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='-----------*--',
drawtext=enable='between(t,12,13)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='------------*-',
drawtext=enable='between(t,13,14)': fontfile=/Library/Fonts/SpaceMono-Bold.otf: fontsize=26: fontcolor=white: x=(w-text_w)/2: y=558: text='-------------*'
"

#########################################
# STITCH TOGETHER


#########################################
# STITCH TOGETHER
for prefix in dep child sex strut;
do

	ffmpeg -i ${prefix}_train_mtds.mp4 -vf "drawbox=x=68: y=588: w=367: h=44: color=LightGrey@0.5: t=fill, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='MTDS': \
		fontsize=30: box=1: boxcolor=0xE0835E@0.7: boxborderw=5: x=80: y=599, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='vs': \
		fontsize=30: x=180: y=605, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Ground Truth': \
		fontsize=30: box=1: boxcolor=0xCCCC00@0.7: boxborderw=5: x=220: y=599, \
		drawbox=x=68: y=588: w=367: h=44: color=black@0.8" -y _${prefix}_train_mtds_ann.mp4

	ffmpeg -i ${prefix}_train_gru.mp4 -vf "drawbox=x=20: y=588: w=454: h=44: color=LightGrey@0.5: t=fill, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='GRU-1L (closed)': \
		fontsize=26: box=1: boxcolor=0x4C9DFF@0.7: boxborderw=3: x=30: y=598, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='vs': \
		fontsize=26: x=245: y=605, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Ground Truth': \
		fontsize=26: box=1: boxcolor=0xCCCC00@0.7: boxborderw=5: x=280: y=600, \
		drawbox=x=20: y=588: w=454: h=44: color=black@0.8" -y _${prefix}_train_gru_ann.mp4

	ffmpeg -i ${prefix}_train_lagr.mp4 -vf "drawbox=x=35: y=588: w=435: h=44: color=LightGrey@0.5: t=fill, \
		drawtext=fontfile=/Library/Fonts/Montserrat-Medium.otf: text='Model side-by-side comparison': \
		fontsize=26: x=45: y=598, \
		drawbox=x=35: y=588: w=435: h=44: color=black@0.8" -y _${prefix}_train_lagr_ann.mp4

ffmpeg -i _${prefix}_train_mtds_ann.mp4 -i _${prefix}_train_gru_ann.mp4 -i _${prefix}_train_lagr_ann.mp4 \
	-filter_complex "[0]pad=iw+10:color=black[left];[1]pad=iw+10:color=black[mid];[left][mid][2]hstack=inputs=3" ${prefix}.mp4;
done

 # CREATING TITLE SLIDES
 ./make_slides.sh

# STITCH ALL
./stitch_all slide01.mp4 slide02.mp4 slide1_depressed.mp4 dep.mp4 slide2_childlike.mp4 child.mp4 \
slide3_sexy.mp4 sex.mp4 slide4_strutting.mp4 strut.mp4 merged.mp4




# MTLopen m
./stitch_all slide01.mp4 slide02.mp4 slide03.mp4 \
slide1.mp4 dep8.mp4 slide2.mp4 dep16.mp4 slide3.mp4 proud8.mp4 \
slide4.mp4 proud16.mp4 slide5.mp4 sexy8.mp4 slide6.mp4 sexy16.mp4 \
slide7.mp4 child8.mp4 slide8.mp4 child16.mp4 merged.mp4

./stitch_all slide01.mp4 slide02.mp4 slide03.mp4 \
slide1.mp4 dep8.mp4 slide2.mp4 dep16.mp4 slide5.mp4 sexy8.mp4 slide6.mp4 sexy16.mp4 \
slide7.mp4 child8.mp4 slide8.mp4 child16.mp4 merged.mp4


# From elsewhere
./stitch_all slide01.mp4 slide02.mp4 slide03.mp4 slide1.mp4 dep8.mp4 slide2.mp4 dep16.mp4 slide5.mp4 sexy8.mp4 slide6.mp4 sexy16.mp4 slide7.mp4 child8.mp4 slide8.mp4 child16.mp4 merged.mp4
./stitch_all slide01.mp4 slide02.mp4 slide1.mp4 angry.mp4 slide2.mp4 child.mp4 slide3.mp4 neut.mp4 merged.mp4
./stitch_all slide01.mp4 slide02.mp4 slide1.mp4 _neut_dep.mp4 slide2.mp4 _dep_child.mp4 slide3.mp4 _child_strut.mp4 slide4.mp4 _strut_sexy.mp4 slide5.mp4 _sexy_proud.mp4 slide6.mp4 _proud_old.mp4 slide7.mp4 _old_neut.mp4 merged.mp4
