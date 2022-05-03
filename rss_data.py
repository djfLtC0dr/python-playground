import requests
from bs4 import BeautifulSoup
import json

class RssData:
    def __init__(self, url: str, query_type: str):
        self.url = url
        self.query_type = query_type
        self.wods_json = self.webscrape_data_rss()

    # scraping function
    def webscrape_data_rss(self) -> str:
        wod_list = []
        try:
            r = requests.get(self.url + self.query_type)
            soup = BeautifulSoup(r.content, features='xml')
            wods = soup.findAll('item') 
            for w in wods:
                pubDate = w.find('pubDate').text
                encoded_content = w.find('encoded')
                for child in encoded_content.children:
                    content = str(child)
                wod = {
                    'pubDate': pubDate,
                    'content': content,
                }
                wod_list.append(wod)
        except Exception as e:
            print('The scraping job failed. See exception: ')
            print(e)
        return json.dumps(wod_list)