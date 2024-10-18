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
browser.get("https://www.facebook.com")

sleep(5)
browser.close()