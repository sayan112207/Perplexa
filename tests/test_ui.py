from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Modify this if the app is hosted elsewhere
# Things to Change:
#If the app is hosted elsewhere, update APP_URL = "http://your-host-url".
APP_URL = "http://localhost:5500"

# Initialize WebDriver
driver = webdriver.Chrome()
driver.get(APP_URL)

# Wait for Streamlit app to load
time.sleep(5)

# Select model
model_dropdown = driver.find_element(By.XPATH, "//select[@aria-label='Select Model']")
model_dropdown.send_keys("Gemini")

# Enter chat prompt
chat_input = driver.find_element(By.XPATH, "//textarea[@aria-label='Type your message here...']")
chat_input.send_keys("Hello, how does this work?")
chat_input.send_keys(Keys.ENTER)

# Wait for response
time.sleep(5)

# Verify response exists
responses = driver.find_elements(By.CLASS_NAME, "stChatMessage")
assert len(responses) > 0, "No response received"

print("UI Test Passed!")

driver.quit()

# Run the UI Test
# streamlit run app.py
# python tests/test_ui.py