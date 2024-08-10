import subprocess
import os
#run this command
'''
ffmpeg -r 24 -pattern_type glob -i '/home/jack/Desktop/Flask_Make_Art/static/archived-images/*.jpg' -c:v libx264 -vf fps=24 -pix_fmt yuv420p -y /home/jack/Desktop/Flask_Make_Art/static/archived-images/archive-images.mp4
'''
cmd='ffmpeg -r 24 -pattern_type glob -i "/home/jack/Desktop/Flask_Make_Art/static/archived-images/*.jpg" -c:v libx264 -vf fps=24 -pix_fmt yuv420p -y /home/jack/Desktop/Flask_Make_Art/static/archived-images/archive-images.mp4'
subprocess.run(cmd,shell=True)

cmd='ffmpeg -r 24 -pattern_type glob -i "/home/jack/Desktop/Flask_Make_Art/static/archived-store/*.png" -c:v libx264 -vf fps=24 -pix_fmt yuv420p -y /home/jack/Desktop/Flask_Make_Art/static/archived-images/archive-store.mp4'
subprocess.run(cmd,shell=True)
#remove all the images
cmd='rm /home/jack/Desktop/Flask_Make_Art/static/archived-images/*.jpg'
subprocess.run(cmd,shell=True)
cmd='rm /home/jack/Desktop/Flask_Make_Art/static/archived-store/*.png'
subprocess.run(cmd,shell=True)