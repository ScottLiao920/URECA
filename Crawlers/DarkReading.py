from bs4 import BeautifulSoup
import urllib3
import os
import warnings
import re
global url_next
global file_no
file_no = 0
folder = "G:/URECA/textsum/CWD/corpus/"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def make_soup(web):
    http = urllib3.PoolManager()
    r = http.request("GET", web, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.103 Safari/537.36'})
    return BeautifulSoup(r.data, 'lxml')


def gettitle(soup):
    title = soup.title.get_text()
    return title


def generate_txt(name, url, date, article, count):
    try:
        name = name.replace("  ", " ").replace("\n", "").replace("\t", "")[0:100]
        new = folder + name + '.txt'
        if(os.path.exists(new)):
            print("File already exists!")
            return count
        else:
            file = open(new, 'w', errors='ignore')
            file.write(url)
            file.write('\n' + date)
            file.write('\n' + name)
            for i in article:
                text = i.get_text().replace('\xa0 ', ' ').encode('utf-8')
                text = text.decode('ascii', 'ignore')  # this is to delete all the /sa0 in unicode
                if not (text[0:8] == "Related:"):
                    file.write('\n' + text)
            file.close()
            count = count + 1
    except OSError:
        print("Illegal file name!")
    return count


url_next = "https://www.darkreading.com/search.asp?q=news&page=1"
n = 1
while url_next:
    try:
        target_list = make_soup(url_next).find("dl", {"class": "results"}).find_all("dt")
        for target in target_list:
            nxt = "https://www.darkreading.com" + target.a["href"]
            print(nxt)
            nxt_soup = make_soup(nxt)
            title = gettitle(nxt_soup)
            date = nxt_soup.find("span", {"class": "smaller gray"}).get_text()
            date = date[0:str(date).find("201")+4]
            content = nxt_soup.find("div", {"id": "article-main"}).find_all("p")
            if content:
                print(generate_txt(title, nxt, date, content, file_no))
            else:
                break
            file_no += 1
        n += 1
        url_next = "https://www.darkreading.com/search.asp?q=news&page=" + str(n)
    except AttributeError:
        print("Need to check content!")
    print("All downloaded!")
    print(url_next)

