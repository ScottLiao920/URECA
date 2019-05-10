import os
from bs4 import BeautifulSoup
import urllib3
global url_next


file_no = 0
folder = "G:/URECA/textsum/CWD/corpus/"


def make_soup(web):
    http = urllib3.PoolManager()
    r = http.request("GET", web, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.103 Safari/537.36'})
    return BeautifulSoup(r.data, 'lxml')


def gettitle(soup):
    title = soup.title.get_text()
    return title


def generate_txt(name, url, date, article, count):
    print(name)
    try:
        name = name.replace("\n", "").replace("\t", "")[0:100]
        new = folder + name +'.txt'
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
    except WindowsError:
        print("Illegal file name!")
    return count


'''def scmagazine(url):
    print(url)
    global file_no
    global url_next
    soup = make_soup(url)
    title = gettitle(soup)
    #print(soup.prettify())
    content = soup.article
    #print(content)
    if(content == None):
        target = soup.findAll("div", {"class": "col-3"})
        count = 0;
        for i in target:
            if(count == 1):
                url_next = i.a["href"]
            count = count + 1
    date = content.time
    if date:
        date = date.get_text()
    else:
        date = 'None'
    article = content.findAll('p')
    file_no = file_no + 1
    url_next = soup.div
    target = soup.findAll("div", {"class": "col-3"})
    count = 0
    path = folder + title.replace("?", " ") + '.txt'
    for i in target:
        url_next = i.a["href"]
        count = count + 1
        if(os.path.exists(path)):
            print("File already exists!")
            continue
        else:
            if article:
                print(generate_txt(title, url, date, article, file_no))
            else:
                print("The layout has changed!")
            scmagazine(url_next)

            print(url_next)'''


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url_next = "https://www.scmagazine.com/home/news/"
while url_next:
    target_list = make_soup(url_next).find("div", {"itemprop": "mainEntityOfPage"}).find_all("article")
    for target in target_list:
        nxt = target.h3.a["href"]
        print(nxt)
        nxt_soup = make_soup(nxt)
        title = gettitle(nxt_soup).replace(" | SC Media", "")
        try:
            date = nxt_soup.time.get_text().replace("				", "").replace("\n", "")
            content = nxt_soup.find("div", {"class": "post-content"}).find_all("p")
            if content:
                print(generate_txt(title, nxt, date, content, file_no))
                file_no += 1
            else:
                print("Lay out change!")
        except AttributeError:
            print("Need to check content!")
    url_next = make_soup(url_next).find("a", {"class": "next page-numbers"})
    if url_next:
        url_next = url_next["href"]
        print(url_next)
    else:
        print("All downloaded!")

