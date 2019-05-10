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


url = "https://thehackernews.com"

def thehackernew(url):
    global file_no
    target = make_soup(url).findAll("div", {"class": "body-post clear"})
    for i in target:
        url_next = make_soup(i.a["href"])
        title = gettitle(url_next)
        illegal = "!@#$%^&*()\"\"{}][:;'?/.,\|™=<>"
        legal =   "                             "
        trantab = str.maketrans(illegal, legal)
        title = title.translate(trantab)
        date = url_next.find("span", {"class": "author"}).get_text()
        new = folder + title + '.txt'
        if os.path.exists(new):
            continue
        else:
            file = open(new, 'w', errors='ignore')
            file.write(i.a["href"])
            file.write('\n' + date)
            file.write('\n' + title)
            for br in url_next.find_all('br'):
                next_s = br.nextSibling
                if not (next_s and isinstance(next_s, NavigableString)):
                    continue
                next2_s = next_s.nextSibling
                if next2_s and isinstance(next2_s, Tag) and next2_s.name == 'br':
                    text = str(next_s).strip()
                    if text:
                        file.write(next_s)
            print(file_no)
            file_no +=1

for page in range(1000):
    thehackernew(url)
    if(make_soup(url).find("span", {"id": 'blog-pager-older-link'})):
        url = make_soup(url).find("span", {"id": 'blog-pager-older-link'}).a["href"]
        print(url)
        page+=1
    else:
        print("All downloaded!")
        break