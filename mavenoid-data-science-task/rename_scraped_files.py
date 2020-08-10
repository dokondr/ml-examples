""" Give scraped files tmore readable names.
    Run it just once.
"""

import os 

def main():
    
    path = './data-science-faq-bot-task-sources/'

    for filename in os.listdir(path): 
        new_name = filename.split('%2F')[-2]+"-"+filename.split('%2F')[-1] 
        print(new_name)
        os.rename(path+filename, path+new_name)
        
if __name__ == '__main__': 
    main() 

