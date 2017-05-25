from bs4 import BeautifulSoup
import pandas as pd
import urllib

titles = []
posted_date = []
description = []
filed = []
tag_list = []
urls = []

acts = pd.DataFrame(columns = ['company','posted_date','date','tags','description','urls'])
url_template = 'https://www.consumerfinance.gov/policy-compliance/enforcement/actions/?page={}'
for n in range(1,20):
    url = url_template.format(n)
    html = urllib.urlopen(url).read()

    soup = BeautifulSoup(html, 'html.parser', from_encoding="utf-8")
    l = soup.find_all(name='article', attrs={'class':'o-post-preview'})

    for i in l:
        title = i.find(name='h3',attrs={'class':'o-post-preview_title'}).text
        title = title.replace('  ','')
        title = title.replace('\n','')
        titles.append(title)
#        urls.append(' ')
        prefix = 'https://www.consumerfinance.gov'
        urls_to_add = []
        for x in i.find_all(name='a'):
                urls_to_add.append(x.get('href'))
                case_link = urls_to_add[0]
                full_link = prefix+str(case_link)
                urls.append(full_link)
        descr = i.find(name='div',attrs={'class':'o-post-preview_description'}).text
        descr = descr.replace('  ','')
        descr = descr.replace('\n','')
        description.append(descr)
        filex = i.find(name='span',attrs={'class':'date meta-header_right'}).text
        filex = filex.replace('  ','')
        filex = filex.replace('\n','')
        filed.append(filex)
        date = i.find(name='span',attrs={'class':'datetime'}).text
        date = date.replace('  ','')
        date = date.replace('\n','')
        posted_date.append(date)
        tags = i.find(name='ul',attrs={'class':'tags_list'}).text
        tags = tags.replace('  ','')
        tags = tags.replace('\n','')
        tag_list.append(tags)

# make become a dataframe
        acts.loc[len(acts)]=[titles[len(acts)],posted_date[len(acts)],filed[len(acts)],
        tag_list[len(acts)],description[len(acts)],urls[len(acts)]]


# acts.to_csv('data/cfpb_actions_2.csv',encoding='utf-8',index=False)
