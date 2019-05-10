from bs4 import BeautifulSoup, NavigableString, Tag
import urllib3
import os
global url_next
global file_no
file_no = 0
folder = "G:/URECA/textsum/CWD/corpus/"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def make_soup(web):
    http = urllib3.PoolManager()
    r = http.request("GET", web)
    return BeautifulSoup(r.data, 'lxml')


def gettitle(soup):
    title = soup.title.get_text()
    return title


def generate_txt(name, url, date, article, count):
    illegal = "!@#%^&*()\"\"{}][:;?/\\|™=<>«"
    legal = "                          "
    trantab = str.maketrans(illegal, legal)
    name = name.translate(trantab).replace("  ", " ").replace("\n", "").replace("\t", "")[0:100]
    new = folder + name + '.txt'
    new.replace("-", " ")
    file = open(new, 'w', errors='ignore')
    file.write(url)
    file.write('\n'+date)
    file.write('\n'+name)
    for i in article:
        text = i.get_text().replace('\xa0 ', ' ').encode('utf-8')
        text = text.decode('ascii', 'ignore')   #this is to delete all the /sa0 in unicode
        if not (text[0:8] == "Related:"):
            file.write('\n' + text)
        #print(text[0:8])
    file.close()
    count = count + 1
    return count


def securityweek(url):
    global  file_no
    url = make_soup(url)
    list = url.find("div", {"class": "panel-pane pane-block pane-views-news-industry-block-1"}).findAll("div", {"class", "views-field-title"})
    count = 0
    for i in list:
        son = "https://www.securityweek.com" + i.a["href"]
        content = make_soup(son)
        title=gettitle(content)[:-19]
        illegal = "!@#$%^&*()\"\"{}][:;'?/.,\|™=<>\t"
        legal = "                              "
        trantab = str.maketrans(illegal, legal)
        title = title.translate(trantab)
        article = content.find("div", {"class": "content clear-block"}).findAll("p")
        if content.find("div", {"class": "submitted"}) == None:
            continue
        date_pattern = str(content.find("div", {"class": "submitted"}).div.get_text())
        date = date_pattern[date_pattern.index('on ') + len("on "):]
        if not (os.path.exists(folder + title + '.txt')):
            if article:
                print(generate_txt(title, son, date, article, file_no))
                file_no += 1
            else:
                print("FUCK! Layout has changed!")
        else:
            print("File already exist!")
        if not(count-9):
            break
        count += 1
url = "https://www.securityweek.com/cybercrime"
securityweek(url)
url = make_soup(url)
url_next = url.find("ul", {"class": "pager"}).find("li", {"class": "pager-next last"}).a["href"]
#print(url_next)
for no in range(1000):
    securityweek("https://www.securityweek.com" + url_next)
    url_next = make_soup("https://www.securityweek.com" + url_next).find("ul", {"class": "pager"}).find("li", {"class": "pager-next last"}).a["href"]
    print("https://www.securityweek.com" + url_next)
    no+=1
