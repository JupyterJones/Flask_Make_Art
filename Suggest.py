import sqlite3
import logging
from sys import argv
# Configure logging
logging.basicConfig(level=logging.DEBUG)

def get_suggestions(search_term):
    try:
        # Create a database connection
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()

        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS dialogue
                     (id INTEGER PRIMARY KEY,
                      search_term TEXT,
                      ChatGPT_PAIR TEXT,
                      ChatGPT_PAIRb BLOB
                      )''')
        conn.commit()

        # Perform operations
        cnt = 0
        DATA = set()
        INDEX = '----SplitHere------'
        with open("App_Sept03", "r") as data:
            Lines = data.read()
            lines = Lines.replace(search_term, INDEX + search_term)
            lines = lines.split(INDEX)
            for line in lines:
                if search_term in line:
                    cnt += 1
                    DATA.add(f'{line[:1200]}')
                    # Insert dialogue pair into the table
                    c.execute("INSERT INTO dialogue (search_term, ChatGPT_PAIR, ChatGPT_PAIRb) VALUES (?, ?, ?)",
                              (search_term, line, line.encode('utf-8')))
                    conn.commit()

        # Close the database connection
        conn.close()
        return DATA
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__=="__main__":
    search_term = argv[1]
    for data in get_suggestions(search_term):
        print(data)
        print('='*50)
