# Configure logging
# Save log data to a file
# ''' Explain the code
'''
I want to log data to a file and read the log file. I want to log data to two files and read the log files.
'''
def logit2(logdata):
    with open("mylog2.txt", "a") as input_file:
        input_file.write(logdata + "\n")
    print("mylog2.txt entry: ", logdata)
    
def readLog1():
    #return the entire log file
    with open("mylog.txt", "r") as input_file:
        for line in input_file:
            print(line)
    #return the last 5 lines of the log file
    LINES = []
    with open("mylog.txt", "r") as input_file:
        lines = input_file.readlines()
        last_lines = lines[-5:]
        for line in last_lines:
            print(line) 
            LINES.append(line)
    return LINES       
    input_file.close()
def readLog2():
    #return the entire log file
    with open("mylog2.txt", "r") as input_file2:
        for line2 in input_file2:
            print(line2)
    #return the last 5 lines of the log file
    LINES2 = []
    with open("mylog2.txt", "r") as input_file2:
        lines2 = input_file2.readlines()
        last_lines2 = lines2[-5:]
        for line2 in last_lines2:
            print(line2) 
            LINES2.append(line2)
    return LINES2       
    input_file.close()    