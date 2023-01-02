import requests
import json


MESSAGE = '2222다른 사람한테 카톡 보내는거는 어떻게 하는지?'


#발행한 토큰 불러오기
with open("./send-katalk/token.json","r") as kakao:
    tokens = json.load(kakao)

print(tokens)

url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

headers={
    "Authorization" : "Bearer " + tokens["access_token"]
}

data = {
       'object_type': 'text',
       'text': MESSAGE,
       'link': {
           'web_url': 'https://developers.kakao.com',
           'mobile_web_url': 'https://developers.kakao.com'
       },
       'button_title': '키워드'
   }
   
data = {'template_object': json.dumps(data)}
response = requests.post(url, headers=headers, data=data)
response.status_code
