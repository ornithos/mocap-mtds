mogrify -background white -alpha remove -alpha off *.png   #thistle
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart -g 30 -bf 2 -c:a aac -b:a 384k -profile:a aac_low -vf "crop=720:480:200:0" output_child.mp4
ffmpeg -i output_sexy.mp4 -i global_sexy.mp4 -filter_complex "[0]pad=iw+10:color=black[left];[left][1]hstack=inputs=2" sexy.mp4

# normal speed. Also removed audio stuff
ffmpeg -i %07d.png -c:v libx264 -preset slow -profile:v high -crf 18 -coder 1 -pix_fmt yuv420p -movflags +faststart -g 30 -bf 2 -r 30 -vf "crop=720:480:200:0" -vf "setpts=(1/2)*PTS"