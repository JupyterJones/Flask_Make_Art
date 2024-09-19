'''This script is used to sort the files in the directory by date.
The function sort_files_by_date(directory) takes a directory as an argument and returns a list of files in the directory sorted by date.
The function sort_files_by_date_desc(directory) takes a directory as an argument and returns a list of files in the directory sorted by date in descending order. keywords: sort files by date desc reverse
'''

def sort_files_by_date(directory):
    import os
    import glob
    return sorted(glob.glob(directory), key=os.path.getmtime)

#search the directory for the files and sort them by date decending
def sort_files_by_date_desc(directory):
    import os
    import glob
    return sorted(glob.glob(directory), key=os.path.getmtime, reverse=True)





directory = 'static/TEXT/*'
files = sort_files_by_date(directory)
print(files)