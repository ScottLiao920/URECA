from bs4 import BeautifulSoup
import urllib3
import os
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


url_next = "https://www.bleepingcomputer.com/"
while url_next:
    target_list = make_soup(url_next).find_all("div", {"class": "bc_latest_news_text"})
    for target in target_list:
        try:
            nxt = target.h4.a["href"]
            nxt_soup = make_soup(nxt)
            title = gettitle(nxt_soup)
            date = nxt_soup.find("li", {"class": "cz-news-date"}).get_text()
            content = nxt_soup.find("div", {"class": "articleBody"}).find_all("p")
            if content:
                print(generate_txt(title, nxt, date, content, file_no))
            else:
                break
            file_no += 1
        except AttributeError:
            print("need to check content!")
    url_next = make_soup(url_next).find("li", {"aria-label": "Next"})
    if url_next:
        url_next = url_next.a["href"]
    else:
        print("All downloaded!")
    print(url_next)