import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep

# 0. Define browser
browser = webdriver.Chrome()

# 1. open Facebook
browser.get("http://facebook.com")

# 2.Load cookie from file

cookies = pickle.load(open("./static/my_cookie.pkl","rb"))
for cookie in cookies:
    browser.add_cookie(cookie)

# 3. Refresh the browser
browser.get("https://www.facebook.com/usuktimes.1st/posts/pfbid038F8uFUM4pxVdpTB5chCe3B8rYDJynQzMUaPtNRmEygaf2vNZfmMPL8izmg2irXffl")
text_post = browser.find_element(By.XPATH, "/html/body/div[1]/div/div/div[1]/div/div[5]/div/div/div[2]/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/div/div/div/div/div/div/div/div/div[13]/div/div/div[3]/div[1]")
print(text_post.text)
sleep(5)
browser.close()