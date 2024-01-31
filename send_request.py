import requests
from concurrent.futures import ThreadPoolExecutor

def send_request(url, data):
    response = requests.post(url, json=data)
    return response.json()

if __name__ == "__main__":
    url = "http://localhost:8000/batch_predict"
    data = [{"text": "hello world"}, \
            {"text": "nihao "}, \
            {"text": "who are you"}, \
            {"text": "hello world"}, \
            {"text": "nihao "}, \
            {"text": "who are you"}, \
            {"text": "hello world"}, \
            {"text": "nihao "}, \
            {"text": "who are you"},\
            {"text": "hello world"}, \
            {"text": "nihao "}, \
            {"text": "who are you"}] 

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda req: send_request(url, req), data)

    for result in results:
        print(result)