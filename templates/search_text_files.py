'''
I want to search a directory static/TEXT for a string in a file and get the line number and the line where the text is found.
'''
import os
import re
import glob
RESULTS = []
def search_text_files(search_text, directory):
    for filename in glob.glob(directory + '/*.txt'):
        with open(filename, 'r') as file:
            file_contents = file.read()
            file_contents = file_contents.split('\`\`\`python')
            for line_number, line in enumerate(file_contents, 1):
                if search_text in line:
                    print(f'Found in {filename} at line {line_number}: {line.strip()}')
                    RESULTS.append(f'Found in {filename} at line {line_number}: {line.strip()}')
    return RESULTS
search_text = 'logit'
#create s filename to save the search_text_files results
filename = search_text+'.txt'
#search the text in the files in the directory                
results = search_text_files(search_text, 'static/TEXT') 
with open(filename, 'w') as file:
    for result in results:
        file.write(result + '\n')   
file.close()                        
    
