import requests
from bs4 import BeautifulSoup
import json


class RssData:
    def __init__(self, url: str, query_type: str, params: str):
        self.url = url
        self.query_type = query_type
        self.params = params
        self.wods_json = self.webscrape_data_rss()

    # scraping function
    def webscrape_data_rss(self) -> str:
        wod_list = []
        try:
            
            headers = {
            'Content-Type' :'application/rss+xml',
            'Authorization': 'Basic Key',
            'Cookie': 'session'
            }        
            r = requests.get(url=self.url, headers=headers, params=self.params)
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