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
            # Added this 5-sec timeout to fix a requests.exceptions.ConnectionError: 
            # ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')) 
            obj_response = requests.get(url=self.url, headers=headers, params=self.params, timeout=5)
            obj_response.raise_for_status()
            if obj_response.status_code == 200:  # 200 for successful call
                soup = BeautifulSoup(obj_response.content, features='xml')
                wods = soup.findAll('item') 
                for w in wods:
                    wod_date = w.find('title').text
                    encoded_content = w.find('encoded')
                    for child in encoded_content.children:
                        wod_details = str(child)
                    wod = {
                        'wod_date': wod_date,
                        'wod_details': wod_details
                    }
                    wod_list.append(wod)
        except requests.exceptions.HTTPError as errh:
            print('The scraping job failed. See exception: ')
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            if errc.args[0].args[1].errno == 104:
                self.webscrape_data_rss() # retry ConnectionResetError(104, 'Connection reset by peer'))
            else:
                print('The scraping job failed. See exception: ')
                print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print('The scraping job failed. See exception: ')
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print('The scraping job failed. See exception: ')
            print ("OOps: Something Else",err)
        finally:
            if len(wod_list) > 0:
                return json.dumps(wod_list)
            else:
                wod = {
                        'wod_date': 'CNX',
                        'wod_details': 'NSTR'
                    }
                wod_list.append(wod)
                return json.dumps(wod_list)