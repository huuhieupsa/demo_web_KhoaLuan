import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep

# Khoi tao Broser
browser = webdriver.Chrome()

# Mở 1 trang web
browser.get("https://www.facebook.com/")

# Điền thông tin
txtUser = browser.find_element(By.ID, "email")
txtUser.send_keys("hieupsaattt")

txtPass = browser.find_element(By.ID, "pass")
txtPass.send_keys("huuhieu09032001")

txtPass.send_keys(Keys.ENTER)

sleep(10)

pickle.dump(browser.get_cookies(), open("my_cookie.pkl","wb"))

browser.close()