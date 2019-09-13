mogrify -background white -alpha remove -alpha off *.png   #thistle
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart -g 30 -bf 2 -c:a aac -b:a 384k -profile:a aac_low -vf "crop=720:480:200:0" output_child.mp4
ffmpeg -i output_sexy.mp4 -i global_sexy.mp4 -filter_complex "[0]pad=iw+10:color=black[left];[left][1]hstack=inputs=2" sexy.mp4

# normal speed. Also removed audio stuff
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart -g 30 -bf 2 -r 30 -vf "crop=720:480:200:0" -vf "setpts=(1/2)*PTS"


### Morphing videos

mogrify -background white -alpha remove -alpha off *.png  
# mogrify -background LightSteelBlue1 -alpha remove -alpha off *.png

ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart \
-g 30 -bf 2 -r 60 -vf "crop=978:550:100:0" -vf "setpts=(1/2)*PTS" output2.mp4


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