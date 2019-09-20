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
