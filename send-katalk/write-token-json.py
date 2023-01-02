import requests
import json

url = 'https://kauth.kakao.com/oauth/token'
client_id = '5cae50e2b8a409aebcee56022a53b81d'
redirect_uri = 'https://example.com/oauth'
code = 'for1bJmvcRt_8dqegX0OqxFFQEbVp5dudNAmOTe7zbQndYPK4VCi43HfJaUXZonHzqvRjgo9dZwAAAGDVi1gsQ'

data = {
    'grant_type':'authorization_code',
    'client_id':client_id,
    'redirect_uri':redirect_uri,
    'code': code,
    }

response = requests.post(url, data=data)
tokens = response.json()

#발행된 토큰 저장
with open("./send-katalk/token.json","w") as kakao:
    json.dump(tokens, kakao)
