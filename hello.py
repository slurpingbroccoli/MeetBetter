import requests

url = "http://172.18.76.85:8000/scrape"
data = {"key": "extremelyrarepictureofaseaspugnar", "url": "https://www.linkedin.com/in/joshua-matte1/"}
response = requests.post(url, json=data)
print(response.json())