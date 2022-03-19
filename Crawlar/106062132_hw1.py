#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
from urllib.request import urlopen
fo = open("input_hw1.txt", "r")
fp = open("106062132_hw1_output.txt", "w+", encoding = "UTF-8")
for line in fo.readlines():
    next_account = line.strip()
    round_num = 4
    address_array = ['0'] * 4
    while(round_num):
        address_array[4 - round_num] = next_account
        url = "https://www.blockchain.com/eth/address/"+ next_account +"?view=standard"
        html = urlopen(url).read()
        soup = BeautifulSoup(html, "html.parser")
        target = soup.find('div','hnfgic-0 blXlQu')('span')
        for i in range(len(target)):
            if(i % 2 == 0 and i != 0):
                tmp = target[i].text + ': ' + target[i+1].text
                print(tmp,file = fp)
        target1 = soup.find_all('div','sc-1fp9csv-0 gkLWFf')
        index = -1
        for i in range(len(target1)):
            if(target1[len(target1) - 1 - i].find('span','sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk sc-85fclk-0 gskKpd') != None):
                index = len(target1)- 1 - i
                break
        if(index == -1):
            print('--------------------------------------------------------------------------',file = fp)
            break
        tmp = 'Date: ' + target1[index].find('span','sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk').text
        print(tmp,file = fp)
        next_account = target1[index].find_all('a','sc-1r996ns-0 dEMBMB sc-1tbyx6t-1 gzmhhS iklhnl-0 dVkJbV')[2].text.strip()
        tmp = 'To: ' + next_account
        print(tmp , file = fp)
        tmp = 'Amount: ' + target1[index].find('span','sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk sc-85fclk-0 gskKpd').text
        print(tmp,file = fp)
        round_num -= 1
        print('--------------------------------------------------------------------------',file = fp)
    index_limit = 3
    for i in range(4):
        if(address_array[i] == '0'):
            index_limit -= 1
    for i in range(index_limit):
        tmp = address_array[i] + " -> "
        print(tmp, end='',file = fp)
    print(address_array[index_limit],file = fp)
    print('--------------------------------------------------------------------------',file = fp)
fp.close()
fo.close()


# In[2]:


get_ipython().system('jupyter nbconvert --to script hw1.ipynb')


# In[ ]:




