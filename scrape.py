from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import requests
import re
import json
import time

base_url = "https://www.yelp.com/biz"
subsequent = "&start=" # subsequent pages represented with 10, 20, 30...

def scrape_reviews(url_endpoint, max_reviews=1000, delay=0):
    url_p1 = base_url + "/" + url_endpoint
    url = url_p1

    reviews = []
    lim = max_reviews * 10 + 1

    for i in range(10, lim, 10):
        print("Scraping page {}...".format(int(i/10)))
        time.sleep(delay)
        try:
            res = requests.get(url)
        except requests.exceptions.ConnectionError:
            break
        soup = BeautifulSoup(res.text, 'html.parser')

        page_reviews = soup.find_all('span', attrs={'class':'raw__373c0__tQAx6', 'lang':'en'})
        reviews.extend([x.text for x in page_reviews])
        url = url_p1 + subsequent + str(i)
        
    # reviews = soup.find_all('span')
    return reviews


def main():
    review_texts = scrape_reviews("anytime-new-york?osq=Restaurants")
    print(len(review_texts))
    print(review_texts[6])
    print(review_texts[30])

if __name__ == "__main__":
    main()
