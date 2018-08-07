# File: pdftogif.py 
# Author(s): Rishikesh Vaishnav, TODO
# Created: 08/07/2018
import sys
import os
import subprocess

filename = sys.argv[1]

SEGMENT_SIZE = 10

page_num_cmd_f = "pdfinfo {0}.pdf | grep Pages | awk '{{print $2}}'"
pdf_clip_cmd_f = "pdftk {0}.pdf cat {1}-{2} output temp{3}.pdf"
gif_create_cmd_f = \
"convert -density 400 -delay 8 -loop 0 -verbose -background white -alpha remove temp{0}.pdf temp{0}.gif"

rmtemp_cmd = "rm temp*"
combine_gifs_cmd_f = "gifsicle {0} > {1}.gif"

numpages = int(subprocess.check_output(page_num_cmd_f.format(filename), \
    shell=True)[:-1])

for i in range(int(numpages / SEGMENT_SIZE) + 1):
    startpage = (i * SEGMENT_SIZE) + 1 
    endpage = (startpage + SEGMENT_SIZE - 1) if startpage \
            <= (numpages - SEGMENT_SIZE - 1) else numpages
    subprocess.call(pdf_clip_cmd_f.format(filename, startpage, endpage, i), \
        shell=True)

    subprocess.call(gif_create_cmd_f.format(i), shell=True)

temp_gif_list = " ".join(["temp{0}.gif".format(j) for j in range(i + 1)])
subprocess.call(combine_gifs_cmd_f.format(temp_gif_list, filename), shell=True)
    
subprocess.call(rmtemp_cmd, shell=True)
